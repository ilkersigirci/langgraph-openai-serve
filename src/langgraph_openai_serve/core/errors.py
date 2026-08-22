"""OpenAI-compatible error response helpers."""

from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai.types.shared import ErrorObject
from starlette.exceptions import HTTPException as StarletteHTTPException

from langgraph_openai_serve.core.logging import exception_type_name, get_logger

if TYPE_CHECKING:
    from starlette.types import HTTPExceptionHandler

logger = get_logger(__name__)


class OpenAIHTTPException(HTTPException):
    """HTTP exception that carries OpenAI error object metadata."""

    def __init__(
        self,
        *,
        status_code: int,
        error: ErrorObject,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail=error.message,
            headers=headers,
        )
        self.error = error


def configure_openai_error_handlers(app: FastAPI) -> None:
    """Install OpenAI-compatible JSON error handlers on a FastAPI app."""
    # Starlette dispatches each handler only for its registered exception class.
    http_handler = cast("HTTPExceptionHandler", openai_http_exception_handler)
    validation_handler = cast(
        "HTTPExceptionHandler",
        openai_request_validation_exception_handler,
    )

    app.add_exception_handler(OpenAIHTTPException, http_handler)
    app.add_exception_handler(
        StarletteHTTPException,
        http_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_handler,
    )
    app.add_exception_handler(Exception, openai_unhandled_exception_handler)


def openai_error_payload(error: ErrorObject) -> dict[str, Any]:
    """Create OpenAI error payload."""
    return {"error": error.model_dump(mode="json")}


async def openai_http_exception_handler(  # ruff: ignore[unused-async]
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle HTTP exceptions."""
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        _log_server_error(request, exc.status_code, exc.__cause__ or exc)

    if isinstance(exc, OpenAIHTTPException):
        error = exc.error
    else:
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        error_type = (
            "server_error"
            if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR
            else "invalid_request_error"
        )
        error = ErrorObject(message=message, type=error_type)

    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error_payload(error),
        headers=getattr(exc, "headers", None),
    )


async def openai_request_validation_exception_handler(  # ruff: ignore[unused-async]
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle validation exceptions."""
    first_error = exc.errors()[0] if exc.errors() else {}
    location = first_error.get("loc", ())
    if not isinstance(location, (tuple, list)):
        location = ()

    parts = [str(part) for part in location if part not in {"body", "query", "path"}]
    param = ".".join(parts) or None
    message = str(first_error.get("msg") or "Invalid request")
    if param:
        message = f"{param}: {message}"

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=openai_error_payload(
            ErrorObject(
                message=message,
                type="invalid_request_error",
                param=param,
            )
        ),
    )


async def openai_unhandled_exception_handler(  # ruff: ignore[unused-async]
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unhandled exceptions."""
    _log_server_error(request, status.HTTP_500_INTERNAL_SERVER_ERROR, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=openai_error_payload(
            ErrorObject(
                message="Internal server error",
                type="server_error",
            )
        ),
    )


def _log_server_error(
    request: Request,
    status_code: int,
    exc: BaseException,
) -> None:
    logger.error(
        "http.request.failed",
        extra={
            "http.request.method": request.method,
            "url.path": request.url.path,
            "http.response.status_code": status_code,
            "error.type": exception_type_name(exc),
        },
        exc_info=(type(exc), exc, exc.__traceback__),
    )
