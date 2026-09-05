"""OpenAI-compatible Responses router."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from openai.types.responses import Response
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.deps import (
    checkpoint_scope_dependency,
    stream_owner_dependency,
)
from langgraph_openai_serve.api.errors import graph_errors
from langgraph_openai_serve.api.models.deps import get_graph_registry_dependency
from langgraph_openai_serve.api.responses.messages import InvalidResponsesInputError
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_responses_request,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import (
    UnsupportedResponsesOutputError,
    generate_response,
)
from langgraph_openai_serve.api.responses.streaming import stream_response
from langgraph_openai_serve.api.streaming import _StreamOwner
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.core.logging import bind_log_context
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.utils import prepare_run

router = APIRouter(tags=["openai"])


@router.post("/responses", response_model=Response)
async def create_response(
    response_request: ResponseCreateRequest,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
    checkpoint_scope: Annotated[str, Depends(checkpoint_scope_dependency)],
    stream_owner: Annotated[
        _StreamOwner,
        Depends(stream_owner_dependency, scope="request"),
    ],
) -> StreamingResponse | Response:
    """Create one stateless OpenAI Response, optionally as an SSE stream."""
    bind_log_context(model=response_request.model, stream=response_request.stream)

    with graph_errors(input_param="input"):
        try:
            graph_request, messages, resume = decode_responses_request(response_request)
        except (UnsupportedResponsesRequestError, InvalidResponsesInputError) as exc:
            raise OpenAIHTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error=ErrorObject(
                    message=str(exc),
                    type="invalid_request_error",
                    param=(
                        exc.param
                        if isinstance(exc, UnsupportedResponsesRequestError)
                        else "input"
                    ),
                ),
            ) from exc
        run = await prepare_run(
            graph_request,
            messages,
            graph_registry,
            resume=resume,
            checkpoint_scope=checkpoint_scope,
        )
        if response_request.stream:
            body = stream_owner.start(
                stream_response(response_request, run),
                run,
            )
            return StreamingResponse(body, media_type="text/event-stream")
        try:
            return await generate_response(response_request, run)
        except UnsupportedResponsesOutputError as exc:
            raise OpenAIHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error=ErrorObject(message=str(exc), type="server_error"),
            ) from exc


__all__ = ["router"]
