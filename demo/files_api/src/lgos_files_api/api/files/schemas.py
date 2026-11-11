"""OpenAI Files wire models that the OpenAI SDK does not define."""

from typing import Literal

from openai.types import FileObject
from pydantic import BaseModel

FilePurpose = Literal["user_data"]
# Deprecated by OpenAI; here it tracks the extraction of the file's content, so
# SDK clients can `wait_for_processing`.
FileStatus = Literal["uploaded", "processed", "error"]


class FileListResponse(BaseModel):
    """Cursor page returned by `GET /v1/files`."""

    object: Literal["list"] = "list"
    data: list[FileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool
