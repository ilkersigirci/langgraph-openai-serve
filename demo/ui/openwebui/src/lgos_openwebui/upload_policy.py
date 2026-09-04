"""Keep Open WebUI file uploads raw for the external Files API."""

from collections.abc import Awaitable, Callable
from importlib import import_module
from typing import TypeAlias, cast
from urllib.parse import parse_qsl, urlencode

Scope: TypeAlias = dict[str, object]
Message: TypeAlias = dict[str, object]
Receive: TypeAlias = Callable[[], Awaitable[Message]]
Send: TypeAlias = Callable[[Message], Awaitable[None]]
ASGIApp: TypeAlias = Callable[[Scope, Receive, Send], Awaitable[None]]


class RawFileUploads:
    """Force Open WebUI's native file upload endpoint to skip processing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and str(scope.get("path", "")).rstrip("/") == "/api/v1/files"
        ):
            scope = _with_processing_disabled(scope)

        await self.app(scope, receive, send)


def _with_processing_disabled(scope: Scope) -> Scope:
    query_string = scope.get("query_string", b"")
    if not isinstance(query_string, bytes):
        msg = "ASGI query_string must be bytes."
        raise TypeError(msg)

    query = [
        (key, value)
        for key, value in parse_qsl(
            query_string.decode("ascii"), keep_blank_values=True
        )
        if key != "process"
    ]
    query.append(("process", "false"))

    updated_scope = dict(scope)
    updated_scope["query_string"] = urlencode(query).encode("ascii")
    return updated_scope


def create_app() -> RawFileUploads:
    """Load the pinned Open WebUI application behind the raw-upload policy."""
    open_webui_app = cast(ASGIApp, import_module("open_webui.main").app)

    return RawFileUploads(open_webui_app)
