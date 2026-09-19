"""FastAPI dependencies local to the Responses route."""

import inspect

from fastapi import Request, status
from openai.types.shared import ErrorObject

from langgraph_openai_serve.background.contracts import BackgroundBackend
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
    request: Request,
    *,
    stream: bool | None = None,
    starting_after: str | None = None,
) -> None:
    """Reject streaming and every supplied cursor before a store read."""
    del starting_after
    if stream:
        message = "Background response retrieval does not support streaming."
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message=message,
                type="invalid_request_error",
                param="stream",
            ),
        )
    if "starting_after" in request.query_params:
        message = "Background response retrieval does not support cursors."
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message=message,
                type="invalid_request_error",
                param="starting_after",
            ),
        )


__all__ = [
    "get_background_backend",
    "get_checkpoint_scope",
    "validate_background_retrieval",
]
