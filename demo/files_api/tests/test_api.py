"""OpenAI Files API behavior, with the content extracted from documents."""

import json
import time
from typing import Any

import pytest
from openai import (
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)

from lgos_files_api.app import create_app
from lgos_files_api.core.settings import settings
from tests.support import BUCKET, docx, processed_file, serve, xlsx

SPREADSHEET_CONTENT = {
    "type": "spreadsheet",
    "version": 1,
    "sheets": [
        {
            "name": "Sales",
            "headers": ["Region", "Total"],
            "rows": [["North", "10"]],
            "chunks": [
                "Sheet: Sales\nFields: Region, Total\n\nRow 1 | Region: North | Total: 10"
            ],
        }
    ],
}


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
            "sales.xlsx",
            xlsx("Sales", [["Region", "Total"], ["North", 10]]),
            SPREADSHEET_CONTENT,
        ),
    ],
    ids=["xberg-document", "text-file", "spreadsheet"],
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


async def test_extraction_cut_short_by_a_restart_completes(s3: Any) -> None:
    # A shutdown during an extraction leaves the original and its queued marker.
    file_id = "file-" + "a" * 32
    s3.put_object(Bucket=BUCKET, Key=f"{file_id}/memo.docx", Body=docx("Resumed"))
    s3.put_object(Bucket=BUCKET, Key=f"extraction/{file_id}.queued", Body=b"{}")

    async with serve(create_app()) as client:
        processed = await processed_file(client, file_id)
        content = await client.files.content(file_id)

    assert processed.status == "processed"
    assert content.json()["text"] == "Resumed\n"


async def test_document_that_stopped_the_service_twice_fails_instead(
    s3: Any,
) -> None:
    # The API already resumed it once, and stopped again before it finished,
    # such as when the document makes it run out of memory.
    file_id = "file-" + "b" * 32
    s3.put_object(Bucket=BUCKET, Key=f"{file_id}/scan.pdf", Body=b"%PDF-1.7")
    s3.put_object(
        Bucket=BUCKET, Key=f"extraction/{file_id}.queued", Body=b'{"resumes": 1}'
    )

    async with serve(create_app()) as client:
        processed = await processed_file(client, file_id)

    assert (processed.status, processed.status_details) == (
        "error",
        "The extraction did not finish.",
    )


@pytest.mark.parametrize(
    ("status_code", "age", "status"),
    [(500, 301, "processed"), (500, 0, "error"), (422, 301, "error")],
    ids=["old-service-failure", "recent-service-failure", "unreadable-file"],
)
async def test_only_service_failures_are_extracted_again_after_a_while(
    s3: Any, status_code: int, age: int, status: str
) -> None:
    file_id = "file-" + "c" * 32
    s3.put_object(Bucket=BUCKET, Key=f"{file_id}/memo.docx", Body=docx("Retried"))
    failure = {
        "status_code": status_code,
        "message": "Earlier failure.",
        "failed_at": time.time() - age,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"extraction/{file_id}.error",
        Body=json.dumps(failure).encode(),
    )

    async with serve(create_app()) as client:
        processed = await processed_file(client, file_id)

    assert processed.status == status


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
