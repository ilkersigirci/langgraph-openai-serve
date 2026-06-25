"""OpenAI-compatible error response helpers."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai.types.shared import ErrorObject
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


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
    app.add_exception_handler(OpenAIHTTPException, openai_http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, openai_http_exception_handler)
    app.add_exception_handler(
        RequestValidationError,
        openai_request_validation_exception_handler,
    )
    app.add_exception_handler(Exception, openai_unhandled_exception_handler)


def openai_error_payload(error: ErrorObject) -> dict[str, Any]:
    return {"error": error.model_dump(mode="json")}


async def openai_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
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


async def openai_request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
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


async def openai_unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled OpenAI-compatible API error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=openai_error_payload(
            ErrorObject(
                message="Internal server error",
                type="server_error",
            )
        ),
    )
