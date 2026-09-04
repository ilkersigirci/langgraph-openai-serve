"""OpenAI Files API contract tests."""

from collections.abc import AsyncIterator
from typing import Literal

import pytest
from fastapi import FastAPI
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


class MemoryFileRepository:
    """Small repository fake for exercising the HTTP contract."""

    def __init__(self) -> None:
        self._files: dict[str, tuple[FileObject, bytes, str]] = {}

    def create(self, upload: FileUpload) -> FileObject:
        file_id = f"file-{len(self._files) + 1}"
        content = upload.body.read()
        stored = FileObject(
            id=file_id,
            bytes=len(content),
            created_at=len(self._files) + 1,
            filename=upload.filename,
            object="file",
            purpose=upload.purpose,
            status="processed",
        )
        self._files[file_id] = (stored, content, upload.content_type)
        return stored

    def list_files(
        self,
        *,
        after: str | None,
        limit: int,
        order: Literal["asc", "desc"],
        purpose: str | None,
    ) -> FilePage:
        files = [stored for stored, _, _ in self._files.values()]
        if purpose is not None:
            files = [stored for stored in files if stored.purpose == purpose]
        files.sort(key=lambda stored: stored.created_at, reverse=order == "desc")
        if after is not None:
            files = files[
                next(
                    (
                        index + 1
                        for index, stored in enumerate(files)
                        if stored.id == after
                    ),
                    len(files),
                ) :
            ]
        return FilePage(data=files[:limit], has_more=len(files) > limit)

    def retrieve(self, file_id: str) -> FileObject:
        try:
            return self._files[file_id][0]
        except KeyError as error:
            raise StoredFileNotFoundError(file_id) from error

    def delete(self, file_id: str) -> FileDeleted:
        self.retrieve(file_id)
        del self._files[file_id]
        return FileDeleted(id=file_id, deleted=True, object="file")

    def content(self, file_id: str) -> FileDownload:
        try:
            _, content, content_type = self._files[file_id]
        except KeyError as error:
            raise StoredFileNotFoundError(file_id) from error
        return FileDownload(
            body=[content],
            content_type=content_type,
            content_length=len(content),
        )


@pytest.fixture
def files_app() -> FastAPI:
    repository: FileRepository = MemoryFileRepository()
    return create_files_app(repository)


@pytest.fixture
async def files_client(files_app: FastAPI) -> AsyncIterator[AsyncOpenAI]:
    async with (
        AsyncClient(
            transport=ASGITransport(app=files_app),
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


async def test_file_lifecycle_matches_openai_client(files_client: AsyncOpenAI) -> None:
    payload = b"\x00\x01\x02"
    uploaded = await files_client.files.create(
        file=("payload.bin", payload, "application/octet-stream"),
        purpose="user_data",
    )

    assert uploaded.id.startswith("file-")
    assert uploaded.filename == "payload.bin"
    assert uploaded.bytes == len(payload)
    assert uploaded.purpose == "user_data"

    retrieved = await files_client.files.retrieve(uploaded.id)
    page = await files_client.files.list()
    response = await files_client.files.content(uploaded.id)

    assert retrieved == uploaded
    assert [file.id for file in page.data] == [uploaded.id]
    assert await response.aread() == payload

    deleted = await files_client.files.delete(uploaded.id)
    assert deleted.id == uploaded.id
    assert deleted.deleted is True

    with pytest.raises(NotFoundError) as exc_info:
        await files_client.files.retrieve(uploaded.id)
    assert exc_info.value.response.json()["error"]["code"] == "file_not_found"


async def test_file_expiration_is_rejected_instead_of_ignored(
    files_client: AsyncOpenAI,
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
    assert (await files_client.files.list()).data == []
