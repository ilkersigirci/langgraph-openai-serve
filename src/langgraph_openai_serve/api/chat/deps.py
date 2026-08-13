"""Dependencies for chat completion routes."""

import inspect
from collections.abc import AsyncIterator

from fastapi import Request

from langgraph_openai_serve.api.chat.utils.streaming import _StreamOwner


async def stream_owner_dependency() -> AsyncIterator[_StreamOwner]:
    owner = _StreamOwner()
    try:
        yield owner
    finally:
        await owner.aclose()


async def checkpoint_scope_dependency(request: Request) -> str:
    value = request.app.state.checkpoint_scope(request)
    if inspect.isawaitable(value):
        value = await value
    return value
