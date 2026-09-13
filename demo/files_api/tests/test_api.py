"""OpenAI Files API contract tests."""

from collections.abc import AsyncIterator
from unittest.mock import Mock, call

import pytest
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI, BadRequestError, NotFoundError
from openai.types import FileDeleted, FileObject

from lgos_files_api import FileRepository, create_files_app
from lgos_files_api.contracts import (
    FileDownload,
    FilePage,
    FileUpload,
    StoredFileNotFoundError,
)


@pytest.fixture
def file_repository() -> Mock:
    return Mock(spec=FileRepository)


@pytest.fixture
async def files_client(file_repository: Mock) -> AsyncIterator[AsyncOpenAI]:
    async with (
        AsyncClient(
            transport=ASGITransport(app=create_files_app(file_repository)),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as client,
    ):
        yield client


async def test_file_routes_translate_sdk_requests_and_repository_results(
    files_client: AsyncOpenAI, file_repository: Mock
) -> None:
    payload = b"\x00\x01\x02"
    stored = FileObject(
        id="file-example",
        bytes=len(payload),
        created_at=1,
        filename="payload.bin",
        object="file",
        purpose="user_data",
        status="processed",
    )

    def create(upload: FileUpload) -> FileObject:
        assert upload.body.read() == payload
        assert upload.size == len(payload)
        assert upload.filename == "payload.bin"
        assert upload.purpose == "user_data"
        assert upload.content_type == "application/octet-stream"
        return stored

    file_repository.create.side_effect = create
    file_repository.retrieve.side_effect = [
        stored,
        StoredFileNotFoundError(stored.id),
    ]
    file_repository.list_files.return_value = FilePage(data=[stored], has_more=False)
    file_repository.content.return_value = FileDownload(
        body=[payload],
        content_type="application/octet-stream",
        content_length=len(payload),
    )
    file_repository.delete.return_value = FileDeleted(
        id=stored.id, deleted=True, object="file"
    )

    uploaded = await files_client.files.create(
        file=("payload.bin", payload, "application/octet-stream"),
        purpose="user_data",
    )

    assert uploaded == stored
    file_repository.create.assert_called_once()

    retrieved = await files_client.files.retrieve(uploaded.id)
    page = await files_client.files.list(limit=1, order="asc", purpose="user_data")
    response = await files_client.files.content(uploaded.id)

    assert retrieved == uploaded
    assert [file.id for file in page.data] == [uploaded.id]
    assert await response.aread() == payload
    file_repository.list_files.assert_called_once_with(
        after=None, limit=1, order="asc", purpose="user_data"
    )
    file_repository.content.assert_called_once_with(uploaded.id)

    deleted = await files_client.files.delete(uploaded.id)
    assert deleted.id == uploaded.id
    assert deleted.deleted is True
    file_repository.delete.assert_called_once_with(uploaded.id)

    with pytest.raises(NotFoundError) as exc_info:
        await files_client.files.retrieve(uploaded.id)
    assert exc_info.value.response.json()["error"]["code"] == "file_not_found"
    assert file_repository.retrieve.call_args_list == [
        call(uploaded.id),
        call(uploaded.id),
    ]


async def test_file_expiration_is_rejected_instead_of_ignored(
    files_client: AsyncOpenAI,
    file_repository: Mock,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await files_client.files.create(
            file=("payload.bin", b"payload"),
            purpose="user_data",
            expires_after={"anchor": "created_at", "seconds": 3600},
        )

    assert exc_info.value.response.json()["error"] == {
        "message": "The expires_after parameter is not supported.",
        "type": "invalid_request_error",
        "param": "expires_after",
        "code": None,
    }
    file_repository.create.assert_not_called()
