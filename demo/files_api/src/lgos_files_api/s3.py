"""S3 repository for the standalone Files service."""

from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from io import SEEK_END
from time import time
from typing import Literal, Protocol, cast
from uuid import uuid4

from botocore.exceptions import ClientError
from openai.types import FileDeleted, FileObject

from lgos_files_api.contracts import (
    FileDownload,
    FilePage,
    FilePurpose,
    FileUpload,
    StoredFileNotFoundError,
)

_CHUNK_SIZE = 64 * 1024
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class _StreamingBody(Protocol):
    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        """Iterate response bytes."""

    def close(self) -> None:
        """Close the response body."""


class _Paginator(Protocol):
    def paginate(self, **kwargs: str) -> Iterable[Mapping[str, object]]:
        """Iterate S3 list pages."""


class _S3Client(Protocol):
    def upload_fileobj(
        self,
        fileobj: object,
        bucket: str,
        key: str,
        extra_args: Mapping[str, object],
    ) -> None:
        """Upload a file-like object."""

    def head_object(self, **kwargs: str) -> Mapping[str, object]:
        """Return object metadata."""

    def get_object(self, **kwargs: str) -> Mapping[str, object]:
        """Return an object body and metadata."""

    def delete_object(self, **kwargs: str) -> Mapping[str, object]:
        """Delete an object."""

    def get_paginator(self, operation_name: str) -> _Paginator:
        """Create an operation paginator."""


@dataclass(frozen=True, slots=True)
class _ListedFile:
    file_id: str
    created_at: int


class S3FileRepository:
    """Persist OpenAI files in an S3-compatible bucket."""

    def __init__(
        self,
        client: object,
        *,
        bucket: str,
        prefix: str = "openai-files",
    ) -> None:
        if not bucket.strip():
            msg = "S3 file bucket must not be empty."
            raise ValueError(msg)
        self._client = cast("_S3Client", client)
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def create(self, upload: FileUpload) -> FileObject:
        """Upload a file with OpenAI metadata."""
        file_id = f"file-{uuid4().hex}"
        created_at = int(time())
        size = self._upload_size(upload)
        upload.body.seek(0)
        self._client.upload_fileobj(
            upload.body,
            self._bucket,
            self._key(file_id),
            {
                "ContentType": upload.content_type,
                "Metadata": {
                    "created-at": str(created_at),
                    "filename": self._encode_filename(upload.filename),
                    "purpose": upload.purpose,
                },
            },
        )
        return self._file_object(
            file_id=file_id,
            size=size,
            created_at=created_at,
            filename=upload.filename,
            purpose=upload.purpose,
        )

    def list_files(
        self,
        *,
        after: str | None,
        limit: int,
        order: Literal["asc", "desc"],
        purpose: str | None,
    ) -> FilePage:
        """List files using S3's native paginator."""
        if purpose not in {None, "user_data"}:
            return FilePage(data=[], has_more=False)

        listed = sorted(
            self._listed_files(),
            key=lambda item: (item.created_at, item.file_id),
            reverse=order == "desc",
        )
        if after is not None:
            cursor = next(
                (
                    index + 1
                    for index, item in enumerate(listed)
                    if item.file_id == after
                ),
                len(listed),
            )
            listed = listed[cursor:]

        files = []
        for item in listed:
            try:
                files.append(self.retrieve(item.file_id))
            except StoredFileNotFoundError:
                continue
            if len(files) > limit:
                break
        return FilePage(data=files[:limit], has_more=len(files) > limit)

    def retrieve(self, file_id: str) -> FileObject:
        """Retrieve metadata for one S3 object."""
        try:
            response = self._client.head_object(
                Bucket=self._bucket,
                Key=self._key(file_id),
            )
        except ClientError as error:
            self._raise_not_found(file_id, error)
            raise
        metadata = self._metadata(response)
        size = response.get("ContentLength")
        if not isinstance(size, int):
            msg = f"S3 object for '{file_id}' has no content length."
            raise TypeError(msg)
        return self._file_object(
            file_id=file_id,
            size=size,
            created_at=self._created_at(response, metadata),
            filename=self._decode_filename(metadata["filename"]),
            purpose=cast("FilePurpose", metadata["purpose"]),
        )

    def delete(self, file_id: str) -> FileDeleted:
        """Delete one S3 object."""
        self.retrieve(file_id)
        self._client.delete_object(Bucket=self._bucket, Key=self._key(file_id))
        return FileDeleted(id=file_id, deleted=True, object="file")

    def content(self, file_id: str) -> FileDownload:
        """Open one S3 object for streaming."""
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=self._key(file_id),
            )
        except ClientError as error:
            self._raise_not_found(file_id, error)
            raise
        body = cast("_StreamingBody", response["Body"])
        content_length = response.get("ContentLength")
        if not isinstance(content_length, int):
            body.close()
            msg = f"S3 object for '{file_id}' has no content length."
            raise TypeError(msg)
        content_type = response.get("ContentType")
        return FileDownload(
            body=self._body_chunks(body),
            content_type=(
                content_type
                if isinstance(content_type, str)
                else "application/octet-stream"
            ),
            content_length=content_length,
        )

    def _key(self, file_id: str) -> str:
        if not self._valid_file_id(file_id):
            raise StoredFileNotFoundError(file_id)
        return f"{self._prefix}/{file_id}" if self._prefix else file_id

    def _listed_files(self) -> Iterator[_ListedFile]:
        prefix = f"{self._prefix}/" if self._prefix else ""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            if not isinstance(contents, list):
                continue
            for item in contents:
                if not isinstance(item, Mapping):
                    continue
                key = item.get("Key")
                last_modified = item.get("LastModified")
                if isinstance(key, str) and isinstance(last_modified, datetime):
                    file_id = key.removeprefix(prefix)
                    if self._valid_file_id(file_id):
                        yield _ListedFile(
                            file_id=file_id,
                            created_at=int(last_modified.timestamp()),
                        )

    @staticmethod
    def _valid_file_id(file_id: str) -> bool:
        value = file_id[5:]
        return file_id.startswith("file-") and value.isascii() and value.isalnum()

    @staticmethod
    def _upload_size(upload: FileUpload) -> int:
        if upload.size is not None:
            return upload.size
        upload.body.seek(0, SEEK_END)
        return upload.body.tell()

    @staticmethod
    def _metadata(response: Mapping[str, object]) -> Mapping[str, str]:
        metadata = response.get("Metadata")
        if not isinstance(metadata, Mapping):
            msg = "S3 object is missing OpenAI file metadata."
            raise TypeError(msg)
        values = {
            key.lower(): value
            for key, value in metadata.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if "filename" not in values or "purpose" not in values:
            msg = "S3 object is missing OpenAI file metadata."
            raise RuntimeError(msg)
        return values

    @staticmethod
    def _created_at(response: Mapping[str, object], metadata: Mapping[str, str]) -> int:
        created_at = metadata.get("created-at")
        if created_at is not None:
            return int(created_at)
        last_modified = response.get("LastModified")
        if isinstance(last_modified, datetime):
            return int(last_modified.timestamp())
        msg = "S3 object is missing its creation time."
        raise RuntimeError(msg)

    @staticmethod
    def _file_object(
        *,
        file_id: str,
        size: int,
        created_at: int,
        filename: str,
        purpose: FilePurpose,
    ) -> FileObject:
        return FileObject(
            id=file_id,
            bytes=size,
            created_at=created_at,
            filename=filename,
            object="file",
            purpose=purpose,
            status="processed",
        )

    @staticmethod
    def _body_chunks(body: _StreamingBody) -> Iterator[bytes]:
        try:
            yield from body.iter_chunks(_CHUNK_SIZE)
        finally:
            body.close()

    @staticmethod
    def _encode_filename(filename: str) -> str:
        return urlsafe_b64encode(filename.encode()).decode()

    @staticmethod
    def _decode_filename(filename: str) -> str:
        return urlsafe_b64decode(filename.encode()).decode()

    @staticmethod
    def _raise_not_found(file_id: str, error: ClientError) -> None:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in _NOT_FOUND_CODES:
            raise StoredFileNotFoundError(file_id) from error
