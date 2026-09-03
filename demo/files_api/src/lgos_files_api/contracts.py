"""Typed wire and repository contracts for stored files."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO, Literal, Protocol

from openai.types import FileDeleted, FileObject

FilePurpose = Literal["user_data"]


class StoredFileNotFoundError(LookupError):
    """Raised when a file ID does not exist in the repository."""


@dataclass(frozen=True, slots=True)
class FileDownload:
    """Streaming file content returned by a repository."""

    body: Iterable[bytes]
    content_type: str
    content_length: int


@dataclass(frozen=True, slots=True)
class FilePage:
    """One cursor page of stored files."""

    data: list[FileObject]
    has_more: bool


@dataclass(frozen=True, slots=True)
class FileUpload:
    """Transport-neutral upload passed to a repository."""

    body: BinaryIO
    filename: str
    content_type: str
    purpose: FilePurpose
    size: int | None = None


class FileRepository(Protocol):
    """Persistence operations required by the Files API."""

    def create(self, upload: FileUpload) -> FileObject:
        """Store an uploaded file."""

    def list_files(
        self,
        *,
        after: str | None,
        limit: int,
        order: Literal["asc", "desc"],
        purpose: str | None,
    ) -> FilePage:
        """List stored files."""

    def retrieve(self, file_id: str) -> FileObject:
        """Retrieve file metadata."""

    def delete(self, file_id: str) -> FileDeleted:
        """Delete a stored file."""

    def content(self, file_id: str) -> FileDownload:
        """Open a stored file for streaming."""
