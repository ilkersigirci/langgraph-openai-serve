"""Pure ASGI middleware for request correlation and error context."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

from langgraph_openai_serve.core.logging import (
    begin_log_context,
    exception_type_name,
    get_logger,
    reset_log_context,
)

logger = get_logger(__name__)
_REQUEST_ID_HEADER = b"x-request-id"
_MAX_REQUEST_ID_LENGTH = 128


class RequestContextMiddleware:
    """Attach a request ID and request context to the mounted LGOS app."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle HTTP requests and pass non-HTTP scopes through unchanged."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        token = begin_log_context(request_id)
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message = _with_request_id_header(message, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logger.exception(
                "http.request.failed",
                extra={
                    **_request_fields(scope, status_code),
                    "error.type": exception_type_name(exc),
                },
            )
            raise
        finally:
            reset_log_context(token)


def _request_id_from_scope(scope: Scope) -> str:
    values = [
        value
        for name, value in scope.get("headers", ())
        if name.lower() == _REQUEST_ID_HEADER
    ]
    if len(values) == 1:
        request_id = values[0].decode("latin-1").strip()
        if _usable_request_id(request_id):
            return request_id
    return str(uuid.uuid4())


def _usable_request_id(request_id: str) -> bool:
    return (
        bool(request_id)
        and len(request_id) <= _MAX_REQUEST_ID_LENGTH
        and request_id.isascii()
        and request_id.isprintable()
    )


def _with_request_id_header(message: Message, request_id: str) -> Message:
    headers = [
        (name, value)
        for name, value in message.get("headers", [])
        if name.lower() != _REQUEST_ID_HEADER
    ]
    headers.append((_REQUEST_ID_HEADER, request_id.encode("latin-1")))
    return {**message, "headers": headers}


def _request_fields(
    scope: Scope,
    status_code: int | None,
) -> dict[str, str | int]:
    fields: dict[str, str | int] = {
        "http.request.method": scope["method"],
        "url.path": scope["path"],
    }
    if status_code is not None:
        fields["http.response.status_code"] = status_code
    return fields
