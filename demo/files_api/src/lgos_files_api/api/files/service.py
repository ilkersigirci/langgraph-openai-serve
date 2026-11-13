"""
The Files API operations, on files stored in S3 only.

Each file is the upload at "{file_id}/{filename}" and the content
extracted from it at "content/{file_id}.json" (by xberg, or natively for text
files and spreadsheets). The content sits outside the file's prefix, where no
filename can reach. Extraction runs in the background; its state is stored in
S3 too (extraction.py).

Every file is user_data, the only purpose accepted on upload.
"""

import re
import time
from dataclasses import dataclass
from io import SEEK_END
from typing import BinaryIO, Literal
from uuid import uuid4

from fastapi import status
from openai.types import FileObject

from lgos_files_api.api.files.extraction import (
    PROCESSED_STATE,
    QUEUED_STATE,
    Extraction,
    Extractions,
    content_key,
    is_supported,
)
from lgos_files_api.core.errors import (
    FilesApiError,
    InvalidRequestError,
    StoredFileNotFoundError,
)
from lgos_files_api.storage import Download, S3Bucket, StoredObject

# A failure of the service, not of the file, is retried after this long.
EXTRACTION_RETRY_AFTER_SECONDS = 300
# OpenAI-style IDs: "file-" and a UUID's 32 hex digits.
FILE_ID_PREFIX = "file-"
FILE_ID = re.compile(rf"{FILE_ID_PREFIX}[0-9a-f]{{32}}")
# S3 keys hold at most 1024 bytes, "{file_id}/" included; MinIO and Versity's
# POSIX backend also cap each path segment at 255 bytes.
MAX_FILENAME_BYTES = 1024 - len(FILE_ID_PREFIX) - 32 - len("/")
MAX_SEGMENT_BYTES = 255


@dataclass(frozen=True, slots=True)
class FileUpload:
    """Transport-neutral upload passed to the service."""

    body: BinaryIO
    filename: str
    content_type: str
    size: int | None = None


class FileService:
    """The Files API operations on one bucket."""

    def __init__(
        self, bucket: S3Bucket, extractions: Extractions, *, max_upload_bytes: int
    ) -> None:
        self._bucket = bucket
        self._extractions = extractions
        self._max_upload_bytes = max_upload_bytes

    async def create(self, upload: FileUpload) -> FileObject:
        size = _upload_size(upload)
        if not size:
            msg = "File is empty."
            raise InvalidRequestError(msg, param="file")
        if size > self._max_upload_bytes:
            msg = "File is too large."
            raise InvalidRequestError(msg, param="file")
        # Measuring an unknown size leaves the stream at its end.
        upload.body.seek(0)

        # Clients percent-encode quotes in multipart filenames; OpenAI decodes them.
        filename = _normalize_filename(upload.filename.replace("%22", '"'))
        file_id = f"{FILE_ID_PREFIX}{uuid4().hex}"
        await self._bucket.upload(
            f"{file_id}/{filename}", upload.body, content_type=upload.content_type
        )
        stored, _ = await self._find(file_id)
        # After the original is stored, so a failed upload leaves no orphan
        # content. The response doesn't wait for it; `content` returns {} until
        # it is done.
        if not is_supported(filename):
            return stored
        await self._extractions.start(file_id, filename)
        return _with_extraction(stored, QUEUED_STATE)

    async def list_files(
        self,
        *,
        after: str | None,
        limit: int,
        order: Literal["asc", "desc"],
        purpose: str | None,
    ) -> tuple[list[FileObject], bool]:
        """
        Return one page of files and whether more follow.

        Lists the whole bucket and paginates in memory on every call; S3 cannot
        list keys by creation time. Fine for thousands of files.
        """
        if purpose not in {None, "user_data"}:
            return [], False
        # Only "{file_id}/{filename}" keys are files.
        stored = [
            item
            for item in await self._bucket.list_objects(FILE_ID_PREFIX)
            if _is_file_id(item.key.partition("/")[0]) and "/" in item.key
        ]
        # Sort on S3's millisecond LastModified: `created_at` is whole seconds,
        # and OpenAI keeps files uploaded within the same second in upload order.
        stored.sort(
            key=lambda item: (item.last_modified, item.key),
            reverse=order == "desc",
        )
        # Files with no stored state read as "processed"; `retrieve` queues them.
        states = await self._extractions.read_all()
        files = [
            _with_extraction(file, states[file.id]) if file.id in states else file
            for file in map(_to_file_object, stored)
        ]
        if after is not None:
            ids = [file.id for file in files]
            if after not in ids:
                raise StoredFileNotFoundError(after, param="after")
            files = files[ids.index(after) + 1 :]
        return files[:limit], len(files) > limit

    async def retrieve(self, file_id: str) -> FileObject:
        file, _ = await self._find(file_id)
        # "processed" means the content is stored; a file whose extraction was
        # never queued or was lost is extracted now.
        return _with_extraction(file, await self._extraction(file))

    async def delete(self, file_id: str) -> None:
        _, key = await self._find(file_id)
        # Content first: if a delete fails, a retry still finds the original.
        await self._extractions.forget(file_id)
        await self._bucket.delete(key)
        # An extraction that checked for the original before it was deleted may
        # have stored content since; one that checks later removes its own.
        await self._extractions.forget(file_id)

    async def content(self, file_id: str, *, raw: bool) -> Download | None:
        """
        Open the file's content, or return None while it is being extracted.

        OpenAI refuses to return user_data files. Documents return the content
        extracted from them instead; other files, such as the generated images
        that clients display, and `raw` requests return the original.
        """
        file, key = await self._find(file_id)
        if raw or not is_supported(file.filename):
            original = await self._bucket.get(key)
            if original is None:  # deleted between the lookup and the read
                raise StoredFileNotFoundError(file_id)
            return original

        content = await self._bucket.get(content_key(file_id))
        if content is not None:
            return content
        state = await self._extraction(file)
        if state.status == "error":
            raise FilesApiError(
                state.details or "Failed to parse the file.",
                status_code=state.status_code or 500,
            )
        # Queued, running, or stored since the lookup above.
        return None

    async def _find(self, file_id: str) -> tuple[FileObject, str]:
        """Return the file and its S3 key."""
        # Only file IDs name files; anything else, such as "content", would
        # list other objects in the bucket.
        stored = (
            await self._bucket.list_objects(f"{file_id}/")
            if _is_file_id(file_id)
            else []
        )
        if not stored:
            raise StoredFileNotFoundError(file_id)
        return _to_file_object(stored[0]), stored[0].key

    async def _extraction(self, file: FileObject) -> Extraction:
        """
        Return the file's extraction, starting one that is due.

        Due are an extraction never queued or lost, and one that failed for a
        reason other than the file, such as storage or a timeout, long enough
        ago; a file the extractors cannot read stays failed.
        """
        if not is_supported(file.filename):
            return PROCESSED_STATE
        state = await self._extractions.read(file.id)
        if state is None or (
            state.status_code is not None
            and state.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
            and state.failed_at is not None
            and time.time() - state.failed_at >= EXTRACTION_RETRY_AFTER_SECONDS
        ):
            await self._extractions.start(file.id, file.filename)
            return QUEUED_STATE
        return state


def _is_file_id(file_id: str) -> bool:
    return FILE_ID.fullmatch(file_id) is not None


def _upload_size(upload: FileUpload) -> int:
    if upload.size is not None:
        return upload.size
    upload.body.seek(0, SEEK_END)
    return upload.body.tell()


def _normalize_filename(filename: str) -> str:
    """
    Return the filename as an S3 key suffix.

    Empty and "." path segments are dropped, as S3 rejects them; ".." is
    refused, so a key cannot leave the file's prefix, and so are names too long
    for an S3 key.
    """
    segments = [segment for segment in filename.split("/") if segment not in {"", "."}]
    normalized = "/".join(segments)
    if (
        not segments
        or ".." in segments
        or any(len(segment.encode()) > MAX_SEGMENT_BYTES for segment in segments)
        or len(normalized.encode()) > MAX_FILENAME_BYTES
    ):
        msg = f"Invalid filename: {filename!r}"
        raise InvalidRequestError(msg, param="file")
    return normalized


def _to_file_object(stored: StoredObject) -> FileObject:
    """Return the file as "processed"; `_with_extraction` sets the real status."""
    file_id, filename = stored.key.split("/", 1)
    return FileObject(
        id=file_id,
        bytes=stored.size,
        created_at=int(stored.last_modified.timestamp()),
        filename=filename,
        object="file",
        purpose="user_data",
        status="processed",
    )


def _with_extraction(file: FileObject, state: Extraction) -> FileObject:
    return file.model_copy(
        update={"status": state.status, "status_details": state.details}
    )
