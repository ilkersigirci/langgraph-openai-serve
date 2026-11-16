"""OpenAI Files API behavior, with the content extracted from documents."""

from pathlib import Path
from typing import Any

import anyio
import pytest
from openai import (
    AsyncOpenAI,
    BadRequestError,
    ConflictError,
    NotFoundError,
    UnprocessableEntityError,
)

from lgos_files_api.api.files import xberg as files_xberg
from lgos_files_api.app import create_app
from lgos_files_api.core.settings import settings
from tests.support import BUCKET, docx, processed_file, serve, xlsx

# One "Orders" sheet written by xlwt: a date cell, which .xls stores as a
# serial number, and an integer, which it stores as a float.
ORDERS_XLS = Path(__file__).parent / "data" / "orders.xls"


def stored_keys(s3: Any) -> list[str]:
    return [
        item["Key"] for item in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])
    ]


@pytest.mark.parametrize(
    ("filename", "document", "content"),
    [
        (
            "memo.docx",
            docx("Quarterly memo", "Revenue grew 12 percent."),
            {
                "type": "text",
                "version": 1,
                "text": "Quarterly memo\n\nRevenue grew 12 percent.\n",
            },
        ),
        (
            "notes.md",
            "# Notes\n\nShip it ✓".encode(),
            {"type": "text", "version": 1, "text": "# Notes\n\nShip it ✓"},
        ),
        (
            "cities.csv",
            b"city,population\nLyon,522250\n",
            {"type": "text", "version": 1, "text": "city,population\nLyon,522250\n"},
        ),
        (
            "deploy.py",
            b"REGION = 'eu-west-1'\n",
            {"type": "text", "version": 1, "text": "REGION = 'eu-west-1'\n"},
        ),
        (
            "memo.rtf",
            rb"{\rtf1\ansi\deff0 {\fonttbl {\f0 Arial;}}\f0 Revenue grew 12 percent.\par}",
            {"type": "text", "version": 1, "text": "Revenue grew 12 percent.\n"},
        ),
        (
            "sales.xlsx",
            xlsx(
                {
                    "Sales": [["Region", "Total"], ["North", 10]],
                    "Empty": [],
                    "Costs": [["Item", "Cost"], ["Rent", 1200.5]],
                }
            ),
            {
                "type": "spreadsheet",
                "version": 1,
                "sheets": [
                    {
                        "name": "Sales",
                        "headers": ["Region", "Total"],
                        "rows": [["North", "10"]],
                        "chunks": [
                            (
                                "Sheet: Sales\nFields: Region, Total\n\n"
                                "Row 1 | Region: North | Total: 10"
                            )
                        ],
                    },
                    {"name": "Empty", "headers": [], "rows": [], "chunks": []},
                    {
                        "name": "Costs",
                        "headers": ["Item", "Cost"],
                        "rows": [["Rent", "1200.5"]],
                        "chunks": [
                            (
                                "Sheet: Costs\nFields: Item, Cost\n\n"
                                "Row 1 | Item: Rent | Cost: 1200.5"
                            )
                        ],
                    },
                ],
            },
        ),
        (
            "orders.xls",
            ORDERS_XLS.read_bytes(),
            {
                "type": "spreadsheet",
                "version": 1,
                "sheets": [
                    {
                        "name": "Orders",
                        "headers": ["Order", "Shipped", "Qty", "Price"],
                        "rows": [["A-1", "2026-03-14", "12", "9.5"]],
                        "chunks": [
                            (
                                "Sheet: Orders\nFields: Order, Shipped, Qty, Price\n\n"
                                "Row 1 | Order: A-1 | Shipped: 2026-03-14 | Qty: 12 "
                                "| Price: 9.5"
                            )
                        ],
                    }
                ],
            },
        ),
    ],
    ids=[
        "xberg-document",
        "text-file",
        "csv-file",
        "code-file",
        "rich-text-document",
        "spreadsheet",
        "legacy-spreadsheet",
    ],
)
async def test_documents_are_processed_into_their_extracted_content(
    files_client: AsyncOpenAI, filename: str, document: bytes, content: object
) -> None:
    uploaded = await files_client.files.create(
        file=(filename, document), purpose="user_data"
    )
    processed = await processed_file(files_client, uploaded.id)
    extracted = await files_client.files.content(uploaded.id)
    raw = await files_client.files.content(uploaded.id, extra_query={"raw": "true"})

    # Extraction runs after the upload returns.
    assert (uploaded.status, processed.status) == ("uploaded", "processed")
    assert extracted.json() == content
    assert raw.content == document


async def test_text_files_are_stored_as_plain_text(files_client: AsyncOpenAI) -> None:
    # Browsers label code and data files inconsistently, and OpenAI's file inputs
    # reject some of their labels, such as this one.
    uploaded = await files_client.files.create(
        file=("deploy.sh", b"echo ok\n", "application/x-shellscript"),
        purpose="user_data",
    )
    raw = await files_client.files.content(uploaded.id, extra_query={"raw": "true"})

    assert raw.response.headers["content-type"].partition(";")[0] == "text/plain"
    assert raw.content == b"echo ok\n"


async def test_unlabelled_uploads_are_labelled_from_their_filename(
    files_client: AsyncOpenAI,
) -> None:
    # Bifrost forwards every upload with this label, which OpenAI's file inputs
    # reject.
    uploaded = await files_client.files.create(
        file=("memo.docx", b"document", "application/octet-stream"),
        purpose="user_data",
    )
    raw = await files_client.files.content(uploaded.id, extra_query={"raw": "true"})

    assert raw.response.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


async def test_unreadable_document_reports_its_failed_extraction(
    files_client: AsyncOpenAI,
) -> None:
    uploaded = await files_client.files.create(
        file=("broken.pdf", b"%PDF-1.7 truncated"), purpose="user_data"
    )
    processed = await processed_file(files_client, uploaded.id)

    assert (processed.status, processed.status_details) == (
        "error",
        "Failed to parse the file.",
    )
    with pytest.raises(UnprocessableEntityError) as exc_info:
        await files_client.files.content(uploaded.id)
    assert exc_info.value.response.json()["error"]["message"] == (
        "Failed to parse the file."
    )


async def test_content_is_refused_until_the_extraction_ends(
    files_client: AsyncOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    extract = files_xberg.extract
    extracting, finish = anyio.Event(), anyio.Event()

    async def held_extract(*args: Any, **kwargs: Any) -> bytes:
        extracting.set()
        await finish.wait()
        return await extract(*args, **kwargs)

    monkeypatch.setattr(files_xberg, "extract", held_extract)
    document = docx("Not yet")
    uploaded = await files_client.files.create(
        file=("memo.docx", document), purpose="user_data"
    )
    with anyio.fail_after(10):
        await extracting.wait()

    with pytest.raises(ConflictError) as exc_info:
        await files_client.files.content(uploaded.id)
    raw = await files_client.files.content(uploaded.id, extra_query={"raw": "true"})
    finish.set()
    await processed_file(files_client, uploaded.id)
    extracted = await files_client.files.content(uploaded.id)

    assert exc_info.value.response.json()["error"]["message"] == (
        f"The content of {uploaded.id} is still being extracted; retrieve the "
        "file until its status is processed."
    )
    assert raw.content == document
    assert extracted.json()["text"] == "Not yet\n"


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("chart.png", "image/png"),
        ("chart.plotly.json", "application/vnd.plotly.v1+json"),
    ],
)
async def test_files_without_extraction_return_their_original_bytes(
    files_client: AsyncOpenAI, filename: str, media_type: str
) -> None:
    # Clients display the files graphs generate from this content.
    payload = b"\x89generated\x00bytes"

    uploaded = await files_client.files.create(
        file=(filename, payload, media_type), purpose="user_data"
    )
    content = await files_client.files.content(uploaded.id)

    assert uploaded.status == "processed"
    assert content.content == payload
    assert content.response.headers["content-type"] == media_type


async def test_deleting_a_file_removes_it_and_its_content(
    files_client: AsyncOpenAI, s3: Any
) -> None:
    uploaded = await files_client.files.create(
        file=("memo.docx", docx("Short-lived")), purpose="user_data"
    )
    await processed_file(files_client, uploaded.id)

    deleted = await files_client.files.delete(uploaded.id)

    assert (deleted.id, deleted.deleted) == (uploaded.id, True)
    with pytest.raises(NotFoundError) as exc_info:
        await files_client.files.retrieve(uploaded.id)
    assert exc_info.value.response.json()["error"] == {
        "message": f"No such File object: {uploaded.id}",
        "type": "invalid_request_error",
        "param": "id",
        "code": None,
        "misalignment": None,
    }
    assert stored_keys(s3) == []


async def test_list_pages_through_every_file_in_either_order(
    files_client: AsyncOpenAI, s3: Any
) -> None:
    # A shared bucket may hold other objects, even ones that look like files.
    s3.put_object(Bucket=BUCKET, Key="file-notes.txt", Body=b"not a file")
    uploaded = {
        (
            await files_client.files.create(
                file=(f"chart-{index}.png", b"png"), purpose="user_data"
            )
        ).id
        for index in range(3)
    }

    first = await files_client.files.list(limit=2)
    rest = await files_client.files.list(limit=2, after=first.data[-1].id)
    oldest_first = await files_client.files.list(order="asc")
    newest_first = [file.id for file in [*first.data, *rest.data]]

    assert (first.has_more, rest.has_more) == (True, False)
    assert sorted(newest_first) == sorted(uploaded)
    assert [file.id for file in oldest_first.data] == newest_first[::-1]
    assert (await files_client.files.list(purpose="assistants")).data == []
    with pytest.raises(NotFoundError) as exc_info:
        await files_client.files.list(after="file-" + "1" * 32)
    assert exc_info.value.response.json()["error"]["param"] == "after"


async def test_list_reports_a_document_without_content_as_uploaded(
    files_client: AsyncOpenAI, s3: Any
) -> None:
    # Its extraction was lost, such as to a restart before it got a slot.
    file_id = "file-" + "d" * 32
    s3.put_object(Bucket=BUCKET, Key=f"{file_id}/memo.docx", Body=docx("Waiting"))

    listed = await files_client.files.list()

    assert [(file.id, file.status) for file in listed.data] == [(file_id, "uploaded")]


@pytest.mark.parametrize(
    ("upload", "param"),
    [
        ({"file": ("report.pdf", b"%PDF"), "purpose": "assistants"}, "purpose"),
        (
            {
                "file": ("report.pdf", b"%PDF"),
                "purpose": "user_data",
                "expires_after": {"anchor": "created_at", "seconds": 3600},
            },
            "expires_after",
        ),
        ({"file": ("empty.pdf", b""), "purpose": "user_data"}, "file"),
        ({"file": ("../escape.pdf", b"%PDF"), "purpose": "user_data"}, "file"),
    ],
    ids=["purpose", "expiration", "empty-file", "parent-path"],
)
async def test_invalid_uploads_are_rejected_before_anything_is_stored(
    files_client: AsyncOpenAI, s3: Any, upload: dict[str, Any], param: str
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await files_client.files.create(**upload)

    assert exc_info.value.response.json()["error"]["param"] == param
    assert stored_keys(s3) == []


async def test_restarts_fail_only_a_document_cut_short_twice(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "MAX_CONCURRENT_EXTRACTIONS", 1)
    extract = files_xberg.extract
    # Set when the extraction holding the only slot starts, once per run.
    holding: list[anyio.Event] = []

    async def endless_extract(*_args: object, **_kwargs: object) -> bytes:
        # Holds the slot until the shutdown, as a long OCR job, or one that
        # runs the service out of memory, would.
        holding[-1].set()
        await anyio.sleep_forever()
        return b""

    monkeypatch.setattr(files_xberg, "extract", endless_extract)
    holding.append(anyio.Event())
    async with serve(create_app()) as client:
        first = await client.files.create(
            file=("first.docx", docx("First")), purpose="user_data"
        )
        with anyio.fail_after(10):
            await holding[-1].wait()
        waiting = [
            await client.files.create(
                file=(f"doc-{index}.docx", docx(f"Doc {index}")), purpose="user_data"
            )
            for index in (2, 3)
        ]

    # Polling clients start the extractions again; the first one holds the slot
    # until the next shutdown too.
    holding.append(anyio.Event())
    async with serve(create_app()) as client:
        await client.files.retrieve(first.id)
        with anyio.fail_after(10):
            await holding[-1].wait()
        for file in waiting:
            await client.files.retrieve(file.id)

    monkeypatch.setattr(files_xberg, "extract", extract)
    async with serve(create_app()) as client:
        outcomes = [
            (processed.status, processed.status_details)
            for processed in [
                await processed_file(client, file.id) for file in (first, *waiting)
            ]
        ]

    assert outcomes == [
        ("error", "The extraction did not finish."),
        ("processed", None),
        ("processed", None),
    ]


@pytest.mark.parametrize(
    ("failure", "outcome"),
    [
        (
            TimeoutError("Extraction timed out after 240000ms"),
            ("error", "Timed out while parsing the file."),
        ),
        # A failure of the service rather than of the file, such as of storage.
        (ConnectionResetError("Connection reset by peer"), ("processed", None)),
    ],
    ids=["timeout", "service-failure"],
)
async def test_only_failures_of_the_file_itself_are_final(
    files_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    outcome: tuple[str, str | None],
) -> None:
    extract = files_xberg.extract
    failures = [failure]

    async def failing_once(*args: Any, **kwargs: Any) -> bytes:
        if failures:
            raise failures.pop()
        return await extract(*args, **kwargs)

    monkeypatch.setattr(files_xberg, "extract", failing_once)
    uploaded = await files_client.files.create(
        file=("memo.docx", docx("Retried")), purpose="user_data"
    )
    processed = await processed_file(files_client, uploaded.id)
    retrieved_again = await files_client.files.retrieve(uploaded.id)

    assert (processed.status, processed.status_details) == outcome
    assert retrieved_again.status == outcome[0]


@pytest.mark.parametrize("size", [2048, 200_000], ids=["part", "whole-body"])
async def test_uploads_over_the_limit_are_rejected_before_anything_is_stored(
    s3: Any, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)

    async with serve(create_app()) as client:
        with pytest.raises(BadRequestError) as exc_info:
            await client.files.create(
                file=("large.txt", b"x" * size), purpose="user_data"
            )

    assert exc_info.value.response.json()["error"]["message"] == "File is too large."
    assert stored_keys(s3) == []
