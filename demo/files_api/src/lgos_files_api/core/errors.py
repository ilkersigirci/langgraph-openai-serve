"""OpenAI-compatible errors and error responses."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai.types.shared import ErrorObject
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class FilesApiError(Exception):
    """A failure rendered as OpenAI's `{"error": {...}}` body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = ErrorObject(
            message=message,
            type=(
                "server_error"
                if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
                else "invalid_request_error"
            ),
            param=param,
            code=code,
        )


class InvalidRequestError(FilesApiError):
    """A request the Files API rejects."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST, param=param)


class StoredFileNotFoundError(FilesApiError):
    """A file ID that names no stored file."""

    def __init__(self, file_id: str, *, param: str = "id") -> None:
        super().__init__(
            f"No such File object: {file_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            param=param,
        )


def configure_openai_error_handlers(app: FastAPI) -> None:
    """Render every failure in the OpenAI error envelope."""
    for exc_class in (
        FilesApiError,
        StarletteHTTPException,
        RequestValidationError,
        Exception,
    ):
        app.add_exception_handler(exc_class, _openai_error_response)


async def _openai_error_response(  # ruff: ignore[unused-async] - Starlette runs sync exception handlers in a threadpool.
    request: Request,
    exc: Exception,
) -> JSONResponse:
    match exc:
        case FilesApiError():
            status_code = exc.status_code
            error = exc.error
        case StarletteHTTPException():
            status_code = exc.status_code
            error = ErrorObject(
                message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                type=(
                    "server_error"
                    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
                    else "invalid_request_error"
                ),
            )
        case RequestValidationError():
            status_code = status.HTTP_400_BAD_REQUEST
            error = _validation_error(exc)
        case _:
            # Starlette re-raises this for the ASGI server to log its traceback.
            logger.error(
                "Unhandled Files API error for %s %s",
                request.method,
                request.url.path,
            )
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            error = ErrorObject(message="Internal server error", type="server_error")
    return JSONResponse(
        status_code=status_code,
        content={"error": error.model_dump(mode="json")},
    )


def _validation_error(exc: RequestValidationError) -> ErrorObject:
    first_error = exc.errors()[0]
    parts = [
        str(part)
        for part in first_error["loc"]
        if part not in {"body", "query", "path"}
    ]
    param = ".".join(parts) or None
    message = first_error["msg"]
    return ErrorObject(
        message=f"{param}: {message}" if param else message,
        type="invalid_request_error",
        param=param,
    )
