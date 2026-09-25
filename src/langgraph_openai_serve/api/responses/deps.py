"""FastAPI dependencies local to the Responses route."""

import inspect
from typing import NoReturn

from fastapi import Request, status
from openai.types.shared import ErrorObject

from langgraph_openai_serve.background import BackgroundBackend
from langgraph_openai_serve.core.errors import OpenAIHTTPException


async def get_checkpoint_scope(request: Request) -> str:
    """Resolve the server-trusted checkpoint scope for one request."""
    value = request.app.state.checkpoint_scope(request)
    if inspect.isawaitable(value):
        value = await value
    return value


def get_background_backend(request: Request) -> BackgroundBackend | None:
    """Return the optional server-injected background backend."""
    return getattr(request.app.state, "background_backend", None)


def validate_background_retrieval(
    *,
    stream: bool | None = None,
    starting_after: str | None = None,
) -> None:
    """Reject streaming and cursors before reading the background engine."""
    if stream:
        _reject_retrieval("stream")
    if starting_after is not None:
        _reject_retrieval("starting_after")


def _reject_retrieval(param: str) -> NoReturn:
    raise OpenAIHTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        error=ErrorObject(
            message=f"Background response retrieval does not support {param}.",
            type="invalid_request_error",
            param=param,
        ),
    )


__all__ = [
    "get_background_backend",
    "get_checkpoint_scope",
    "validate_background_retrieval",
]
