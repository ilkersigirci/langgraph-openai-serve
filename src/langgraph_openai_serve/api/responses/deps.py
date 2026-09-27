"""FastAPI dependencies local to the Responses route."""

import inspect

from fastapi import Request

from langgraph_openai_serve.background import BackgroundBackend
from langgraph_openai_serve.core.errors import (
    GraphError,
    InvalidRequestError,
)


async def get_checkpoint_scope(request: Request) -> str:
    """Resolve the server-trusted checkpoint scope for one request."""
    value = request.app.state.checkpoint_scope(request)
    if inspect.isawaitable(value):
        value = await value
    if not value:
        msg = "checkpoint_scope must resolve to a non-empty server-trusted string."
        raise GraphError(msg)
    return value


def get_background_backend(request: Request) -> BackgroundBackend | None:
    """Return the optional server-injected background backend."""
    return request.app.state.background_backend


def validate_background_retrieval(
    *,
    stream: bool | None = None,
    starting_after: str | None = None,
) -> None:
    """Reject streaming and cursors before reading the background engine."""
    for param, value in (("stream", stream), ("starting_after", starting_after)):
        if value:
            msg = f"Background response retrieval does not support {param}."
            raise InvalidRequestError(msg, param=param)


__all__ = [
    "get_background_backend",
    "get_checkpoint_scope",
    "validate_background_retrieval",
]
