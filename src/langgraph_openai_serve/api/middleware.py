"""Pure ASGI middleware for request correlation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from starlette.datastructures import Headers, MutableHeaders

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

from langgraph_openai_serve.core.logging import (
    begin_log_context,
    reset_log_context,
)

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

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_log_context(token)


def _request_id_from_scope(scope: Scope) -> str:
    values = Headers(scope=scope).getlist("X-Request-ID")
    if len(values) == 1:
        request_id = values[0].strip()
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
