"""
The Files API operations, on files stored in S3 only.

Each file is the upload at "{file_id}/{filename}" and the content
extracted from it at "content/{file_id}.json" (by xberg, or as they are for
text files). The content sits outside the file's prefix, where no filename can
reach. Extraction runs in the background; its state is stored in S3 too
(extraction.py).

Every file is user_data, the only purpose accepted on upload.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import SEEK_END
from typing import BinaryIO, Literal
from uuid import uuid4

from openai.types import FileObject

from lgos_files_api.api.files import native
from lgos_files_api.api.files.extraction import (
    PROCESSED,
    UPLOADED,
    Extraction,
    Extractions,
    content_key,
    is_supported,
)
from lgos_files_api.core.errors import (
    ContentNotReadyError,
    FilesApiError,
    InvalidRequestError,
    StoredFileNotFoundError,
)
from lgos_files_api.storage import Download, S3Bucket, StoredObject

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
        # Clients label text, data, and code files inconsistently, and OpenAI's
        # file inputs, to which graphs forward originals, reject some of those
        # labels but take every text file as plain text.
        content_type = (
            "text/plain" if native.is_supported(filename) else upload.content_type
        )
        await self._bucket.upload(
            f"{file_id}/{filename}", upload.body, content_type=content_type
        )
        stored, _ = await self._find(file_id)
        # After the original is stored, so a failed upload leaves no orphan
        # content. The response doesn't wait for the extraction.
        if not is_supported(filename):
            return stored
        self._extractions.start(file_id, filename)
        return _with_extraction(stored, UPLOADED)

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
        # OpenAI keeps files uploaded within the same second in upload order.
        # `created_at` is whole seconds, but MinIO and Versity list LastModified
        # in milliseconds; AWS lists whole seconds, so there such files fall
        # back to ID order.
        stored.sort(
            key=lambda item: (item.last_modified, item.key),
            reverse=order == "desc",
        )
        states = await self._extractions.read_all()
        files = [_listed(file, states) for file in map(_to_file_object, stored)]
        if after is not None:
            ids = [file.id for file in files]
            if after not in ids:
                raise StoredFileNotFoundError(after, param="after")
            files = files[ids.index(after) + 1 :]
        return files[:limit], len(files) > limit

    async def retrieve(self, file_id: str) -> FileObject:
        file, _ = await self._find(file_id)
        return _with_extraction(file, await self._extraction(file))

    async def delete(self, file_id: str) -> None:
        _, key = await self._find(file_id)
        # The original first: an extraction that ends later finds it gone and
        # removes its own outcome.
        await self._bucket.delete(key)
        await self._extractions.forget(file_id)

    async def content(self, file_id: str, *, raw: bool) -> Download:
        """
        Open the file's content, refused until its extraction has ended.

        OpenAI refuses to return user_data files. Documents return the content
        extracted from them instead; other files, such as the generated images
        that clients display, and `raw` requests return the original.
        """
        file, key = await self._find(file_id)
        if not raw and is_supported(file.filename):
            state = await self._extraction(file)
            if state.status == "error":
                raise FilesApiError(
                    state.details or "Failed to parse the file.",
                    status_code=state.status_code or 500,
                )
            if state.status == "uploaded":
                raise ContentNotReadyError(file_id)
            key = content_key(file_id)
        download = await self._bucket.get(key)
        if download is None:  # deleted between the lookup and the read
            raise StoredFileNotFoundError(file_id)
        return download

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
        """Return the file's extraction, starting it if none runs and none ended."""
        if not is_supported(file.filename):
            return PROCESSED
        state = await self._extractions.read(file.id)
        if state is None:
            self._extractions.start(file.id, file.filename)
            return UPLOADED
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


def _listed(file: FileObject, states: Mapping[str, Extraction]) -> FileObject:
    """Return a listed file with its status; listing starts no extraction."""
    if file.id in states:
        return _with_extraction(file, states[file.id])
    # A document without content or an error is still being extracted, or its
    # next retrieve starts the extraction again.
    return _with_extraction(file, UPLOADED) if is_supported(file.filename) else file


def _with_extraction(file: FileObject, state: Extraction) -> FileObject:
    return file.model_copy(
        update={"status": state.status, "status_details": state.details}
    )
