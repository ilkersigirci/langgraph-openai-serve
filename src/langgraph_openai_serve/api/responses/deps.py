"""FastAPI dependencies local to the Responses route."""

import inspect

from fastapi import Request


async def get_checkpoint_scope(request: Request) -> str:
    """Resolve the server-trusted checkpoint scope for one request."""
    value = request.app.state.checkpoint_scope(request)
    if inspect.isawaitable(value):
        value = await value
    return value


__all__ = ["get_checkpoint_scope"]
