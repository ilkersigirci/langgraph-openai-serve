"""OpenAI-compatible Files routes and error responses."""

import logging
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn, cast

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types import FileDeleted, FileObject
from openai.types.shared import ErrorObject
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from lgos_files_api.contracts import (
    FilePurpose,
    FileRepository,
    FileUpload,
    StoredFileNotFoundError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])

if TYPE_CHECKING:
    from starlette.types import HTTPExceptionHandler


class FileListResponse(BaseModel):
    """Cursor page returned by `GET /v1/files`."""

    object: Literal["list"] = "list"
    data: list[FileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool


class OpenAIHTTPException(HTTPException):
    """HTTP exception carrying OpenAI error metadata."""

    def __init__(self, *, status_code: int, error: ErrorObject) -> None:
        super().__init__(status_code=status_code, detail=error.message)
        self.error = error


def configure_openai_error_handlers(app: FastAPI) -> None:
    """Install OpenAI-compatible error handlers."""
    # Starlette dispatches each handler only for its registered exception class.
    http_handler = cast("HTTPExceptionHandler", openai_http_exception_handler)
    validation_handler = cast(
        "HTTPExceptionHandler",
        openai_request_validation_exception_handler,
    )
    app.add_exception_handler(OpenAIHTTPException, http_handler)
    app.add_exception_handler(StarletteHTTPException, http_handler)
    app.add_exception_handler(
        RequestValidationError,
        validation_handler,
    )
    app.add_exception_handler(Exception, openai_unhandled_exception_handler)


def _get_file_repository(request: Request) -> FileRepository:
    return cast("FileRepository", request.app.state.file_repository)


def _raise_file_not_found(file_id: str, error: Exception) -> NoReturn:
    raise OpenAIHTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        error=ErrorObject(
            message=f"File '{file_id}' not found.",
            type="invalid_request_error",
            param="file_id",
            code="file_not_found",
        ),
    ) from error


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
    raise OpenAIHTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        error=ErrorObject(
            message="The expires_after parameter is not supported.",
            type="invalid_request_error",
            param="expires_after",
        ),
    )


@router.post("/files", response_model_exclude_none=True)
def create_file(
    file: Annotated[UploadFile, File()],
    purpose: Annotated[FilePurpose, Form()],
    repository: Annotated[FileRepository, Depends(_get_file_repository)],
    _expiration: Annotated[None, Depends(_reject_unsupported_expiration)],
) -> FileObject:
    """Upload a file to the configured repository."""
    return repository.create(
        FileUpload(
            body=file.file,
            filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
            purpose=purpose,
            size=file.size,
        )
    )


@router.get("/files", response_model_exclude_none=True)
def list_files(
    repository: Annotated[FileRepository, Depends(_get_file_repository)],
    after: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 10_000,
    order: Literal["asc", "desc"] = "desc",
    purpose: str | None = None,
) -> FileListResponse:
    """List files in the configured repository."""
    page = repository.list_files(
        after=after,
        limit=limit,
        order=order,
        purpose=purpose,
    )
    return FileListResponse(
        data=page.data,
        first_id=page.data[0].id if page.data else None,
        last_id=page.data[-1].id if page.data else None,
        has_more=page.has_more,
    )


@router.get("/files/{file_id}", response_model_exclude_none=True)
def retrieve_file(
    file_id: str,
    repository: Annotated[FileRepository, Depends(_get_file_repository)],
) -> FileObject:
    """Retrieve metadata for one file."""
    try:
        return repository.retrieve(file_id)
    except StoredFileNotFoundError as error:
        _raise_file_not_found(file_id, error)


@router.delete("/files/{file_id}")
def delete_file(
    file_id: str,
    repository: Annotated[FileRepository, Depends(_get_file_repository)],
) -> FileDeleted:
    """Delete one file."""
    try:
        return repository.delete(file_id)
    except StoredFileNotFoundError as error:
        _raise_file_not_found(file_id, error)


@router.get("/files/{file_id}/content", response_class=StreamingResponse)
def retrieve_file_content(
    file_id: str,
    repository: Annotated[FileRepository, Depends(_get_file_repository)],
) -> StreamingResponse:
    """Stream the original file bytes."""
    try:
        download = repository.content(file_id)
    except StoredFileNotFoundError as error:
        _raise_file_not_found(file_id, error)
    return StreamingResponse(
        download.body,
        media_type=download.content_type,
        headers={"Content-Length": str(download.content_length)},
    )


async def openai_http_exception_handler(
    _request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Render HTTP failures in the OpenAI error envelope."""
    if isinstance(exc, OpenAIHTTPException):
        error = exc.error
    else:
        error = ErrorObject(
            message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            type=(
                "server_error"
                if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
                else "invalid_request_error"
            ),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error.model_dump(mode="json")},
    )


async def openai_request_validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Render request validation failures as OpenAI bad requests."""
    first_error = exc.errors()[0] if exc.errors() else {}
    location = first_error.get("loc", ())
    if not isinstance(location, (tuple, list)):
        location = ()
    parts = [str(part) for part in location if part not in {"body", "query", "path"}]
    param = ".".join(parts) or None
    message = str(first_error.get("msg") or "Invalid request")
    if param:
        message = f"{param}: {message}"
    error = ErrorObject(
        message=message,
        type="invalid_request_error",
        param=param,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": error.model_dump(mode="json")},
    )


async def openai_unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Hide internal exception details behind a stable error response."""
    logger.exception(
        "Unhandled Files API error for %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    error = ErrorObject(message="Internal server error", type="server_error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": error.model_dump(mode="json")},
    )
