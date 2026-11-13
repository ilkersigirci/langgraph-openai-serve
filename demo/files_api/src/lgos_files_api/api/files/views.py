"""OpenAI-compatible Files routes."""

from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types import FileDeleted, FileObject

from lgos_files_api.api.files.schemas import FileListResponse, FilePurpose
from lgos_files_api.api.files.service import FileService, FileUpload
from lgos_files_api.core.errors import InvalidRequestError

router = APIRouter(prefix="/v1/files", tags=["files"])


def _file_service(request: Request) -> FileService:
    return cast("FileService", request.app.state.files)


Files = Annotated[FileService, Depends(_file_service)]


def _reject_unsupported_expiration(
    expires_after: Annotated[str | None, Form()] = None,
    expires_after_anchor: Annotated[
        str | None,
        Form(alias="expires_after[anchor]"),
    ] = None,
    expires_after_seconds: Annotated[
        str | None,
        Form(alias="expires_after[seconds]"),
    ] = None,
) -> None:
    if all(
        value is None
        for value in (expires_after, expires_after_anchor, expires_after_seconds)
    ):
        return
    msg = "The expires_after parameter is not supported."
    raise InvalidRequestError(msg, param="expires_after")


@router.post("", response_model_exclude_none=True)
async def create_file(
    file: Annotated[UploadFile, File()],
    purpose: Annotated[FilePurpose, Form()],  # ruff: ignore[unused-function-argument] - FastAPI validates it; user_data is the only purpose.
    files: Files,
    _expiration: Annotated[None, Depends(_reject_unsupported_expiration)],
) -> FileObject:
    """Upload a file; the content of a document is extracted in the background."""
    return await files.create(
        FileUpload(
            body=file.file,
            filename=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
            size=file.size,
        )
    )


@router.get("", response_model_exclude_none=True)
async def list_files(
    files: Files,
    after: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 10_000,
    order: Literal["asc", "desc"] = "desc",
    purpose: str | None = None,
) -> FileListResponse:
    page, has_more = await files.list_files(
        after=after, limit=limit, order=order, purpose=purpose
    )
    return FileListResponse(
        data=page,
        first_id=page[0].id if page else None,
        last_id=page[-1].id if page else None,
        has_more=has_more,
    )


@router.get("/{file_id}", response_model_exclude_none=True)
async def retrieve_file(file_id: str, files: Files) -> FileObject:
    """Retrieve a file; its status tracks the extraction of its content."""
    return await files.retrieve(file_id)


@router.delete("/{file_id}")
async def delete_file(file_id: str, files: Files) -> FileDeleted:
    """Delete a file and the content extracted from it."""
    await files.delete(file_id)
    return FileDeleted(id=file_id, deleted=True, object="file")


@router.get("/{file_id}/content", response_class=StreamingResponse)
async def retrieve_file_content(
    file_id: str,
    files: Files,
    *,
    raw: Annotated[
        bool,
        Query(description="Return the original upload instead of extracted content."),
    ] = False,
) -> Response:
    """Return a document's extracted content, `{}` until it is ready, or the file."""
    content = await files.content(file_id, raw=raw)
    if content is None:
        return JSONResponse({})
    # Starlette iterates the synchronous body in a worker thread.
    return StreamingResponse(
        content.chunks(),
        media_type=content.content_type,
        headers={"Content-Length": str(content.content_length)},
    )
