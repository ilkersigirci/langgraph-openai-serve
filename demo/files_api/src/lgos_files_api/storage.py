"""An S3-compatible bucket, with boto3's blocking calls run in worker threads."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import IO, TYPE_CHECKING

from anyio import to_thread
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from botocore.response import StreamingBody
    from mypy_boto3_s3 import S3Client

_CHUNK_SIZE = 1024 * 1024
# HEAD errors have no body, so botocore uses the HTTP status as their code.
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


@dataclass(frozen=True, slots=True)
class StoredObject:
    """One object of a listing."""

    key: str
    size: int
    last_modified: datetime


@dataclass(frozen=True, slots=True)
class Download:
    """An object being read, with the content type it was stored with."""

    body: StreamingBody
    content_type: str
    content_length: int

    def chunks(self) -> Iterator[bytes]:
        """
        Stream the body, then close it; iterate it off the event loop.

        Yields:
            The body's chunks.

        """
        try:
            yield from self.body.iter_chunks(_CHUNK_SIZE)
        finally:
            self.body.close()

    async def read(self) -> bytes:
        """Read the whole body in a worker thread and close it."""
        try:
            return await to_thread.run_sync(self.body.read)
        finally:
            self.body.close()


class S3Bucket:
    """The bucket that stores the files, their content, and its extraction state."""

    def __init__(self, client: S3Client, name: str) -> None:
        if not name.strip():
            msg = "S3 file bucket must not be empty."
            raise ValueError(msg)
        self._client = client
        self._name = name

    async def upload(self, key: str, fileobj: IO[bytes], *, content_type: str) -> None:
        """Upload a file-like object; boto3 switches to multipart for large files."""
        await to_thread.run_sync(
            partial(
                self._client.upload_fileobj,
                fileobj,
                self._name,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        )

    async def put(self, key: str, body: bytes, *, content_type: str) -> None:
        await to_thread.run_sync(
            partial(
                self._client.put_object,
                Bucket=self._name,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        )

    async def list_objects(self, prefix: str) -> list[StoredObject]:
        """Return every object under `prefix`."""

        def list_pages() -> list[StoredObject]:
            paginator = self._client.get_paginator("list_objects_v2")
            return [
                StoredObject(
                    key=item["Key"],
                    size=item["Size"],
                    last_modified=item["LastModified"],
                )
                for page in paginator.paginate(Bucket=self._name, Prefix=prefix)
                for item in page.get("Contents", [])
            ]

        return await to_thread.run_sync(list_pages)

    async def get(self, key: str) -> Download | None:
        """Open an object for reading, or return None if it does not exist."""
        try:
            response = await to_thread.run_sync(
                partial(self._client.get_object, Bucket=self._name, Key=key)
            )
        except ClientError as error:
            if _is_not_found(error):
                return None
            raise
        return Download(
            body=response["Body"],
            content_type=response.get("ContentType") or "application/octet-stream",
            content_length=response["ContentLength"],
        )

    async def exists(self, key: str) -> bool:
        try:
            await to_thread.run_sync(
                partial(self._client.head_object, Bucket=self._name, Key=key)
            )
        except ClientError as error:
            if _is_not_found(error):
                return False
            raise
        return True

    async def delete(self, key: str) -> None:
        """Delete an object; S3 also succeeds for missing keys."""
        await to_thread.run_sync(
            partial(self._client.delete_object, Bucket=self._name, Key=key)
        )


def _is_not_found(error: ClientError) -> bool:
    return str(error.response.get("Error", {}).get("Code", "")) in _NOT_FOUND_CODES
