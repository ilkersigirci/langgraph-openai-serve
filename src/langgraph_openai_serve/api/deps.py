"""Dependencies shared by OpenAI-compatible API routes."""

import inspect
from collections.abc import AsyncIterator

from fastapi import Request

from langgraph_openai_serve.api.streaming import _StreamOwner


async def checkpoint_scope_dependency(request: Request) -> str:
    """Resolve the server-trusted checkpoint scope for one request."""
    value = request.app.state.checkpoint_scope(request)
    if inspect.isawaitable(value):
        value = await value
    return value


async def stream_owner_dependency() -> AsyncIterator[_StreamOwner]:
    """
    Manage the streaming producer owned by one request.

    Yields:
        The request-scoped stream owner.

    """
    owner = _StreamOwner()
    try:
        yield owner
    finally:
        await owner.aclose()


__all__ = ["checkpoint_scope_dependency", "stream_owner_dependency"]
