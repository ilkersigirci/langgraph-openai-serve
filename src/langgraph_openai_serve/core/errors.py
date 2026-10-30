"""OpenAI-compatible error responses."""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.exceptions import ContextOverflowError
from openai.types.shared import ErrorObject
from starlette.exceptions import HTTPException

from langgraph_openai_serve.core.logging import (
    exception_type_name,
    get_log_context,
    get_logger,
)

logger = get_logger(__name__)


class InvalidRequestError(Exception):
    """A request rejected with an OpenAI ``invalid_request_error`` envelope."""

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.param = param
        self.code = code
        self.status_code = status_code


class GraphError(RuntimeError):
    """Raised when a graph's configuration or output cannot be served."""


def configure_openai_error_handlers(app: FastAPI) -> None:
    """Return every error from ``app`` as an OpenAI-compatible JSON envelope."""
    for exc_class in (
        InvalidRequestError,
        GraphError,
        ContextOverflowError,
        HTTPException,
        RequestValidationError,
        Exception,
    ):
        app.add_exception_handler(exc_class, _openai_error_response)


def openai_error_payload(error: ErrorObject) -> dict[str, Any]:
    """Create OpenAI error payload."""
    return {"error": error.model_dump(mode="json")}


async def _openai_error_response(  # ruff: ignore[unused-async] - Starlette runs sync exception handlers in a threadpool.
    request: Request,
    exc: Exception,
) -> JSONResponse:
    headers = None
    match exc:
        case InvalidRequestError():
            status_code = exc.status_code
            error = ErrorObject(
                message=str(exc),
                type="invalid_request_error",
                param=exc.param,
                code=exc.code,
            )
        case RequestValidationError():
            status_code = status.HTTP_400_BAD_REQUEST
            error = _validation_error(exc)
        case ContextOverflowError():
            # Answer as OpenAI does: a retryable 500 would make OpenAI clients
            # rerun the whole graph into the same limit.
            status_code = status.HTTP_400_BAD_REQUEST
            error = ErrorObject(
                message="The input exceeds the model's context window.",
                type="invalid_request_error",
                code="context_length_exceeded",
            )
        case HTTPException() if exc.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            status_code = exc.status_code
            error = ErrorObject(message=str(exc.detail), type="invalid_request_error")
            headers = exc.headers
        case _:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            _log_server_error(request, status_code, exc)
            error = ErrorObject(message="Internal server error", type="server_error")
    return JSONResponse(
        status_code=status_code,
        content=openai_error_payload(error),
        headers=headers,
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


def _log_server_error(
    request: Request,
    status_code: int,
    exc: BaseException,
) -> None:
    # Starlette re-raises what only its catch-all handler catches, so the ASGI
    # server logs that traceback; the note ties it to this record. GraphError and
    # HTTPException stop here, so this record carries their traceback.
    reraised = not isinstance(exc, (GraphError, HTTPException))
    if reraised:
        exc.add_note(f"request_id: {get_log_context().get('request_id')}")
    logger.error(
        "http.request.failed",
        extra={
            "http.request.method": request.method,
            "url.path": request.url.path,
            "http.response.status_code": status_code,
            "error.type": exception_type_name(exc),
        },
        exc_info=None if reraised else exc,
    )
