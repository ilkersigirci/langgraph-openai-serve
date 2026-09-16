"""OpenAI-compatible Responses router."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from openai.types.responses import Response
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.deps import get_graph_registry, get_stream_owner
from langgraph_openai_serve.api.errors import graph_errors
from langgraph_openai_serve.api.responses import service as responses_service
from langgraph_openai_serve.api.responses.deps import get_checkpoint_scope
from langgraph_openai_serve.api.responses.messages import InvalidResponsesInputError
from langgraph_openai_serve.api.responses.output import UnsupportedResponsesOutputError
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.streaming import StreamOwner
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.core.logging import bind_log_context
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

router = APIRouter(tags=["openai"])


@router.post("/responses", response_model=Response)
async def create_response(
    response_request: ResponseCreateRequest,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry)],
    checkpoint_scope: Annotated[str, Depends(get_checkpoint_scope)],
    stream_owner: Annotated[
        StreamOwner,
        Depends(get_stream_owner, scope="request"),
    ],
) -> StreamingResponse | Response:
    """Create one stateless OpenAI Response, optionally as an SSE stream."""
    bind_log_context(model=response_request.model, stream=response_request.stream)

    with graph_errors(input_param="input"):
        try:
            run = await responses_service.prepare_response_run(
                response_request,
                graph_registry,
                checkpoint_scope=checkpoint_scope,
            )
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
        if response_request.stream:
            body = stream_owner.start(
                responses_service.stream_response(response_request, run),
                run,
            )
            return StreamingResponse(body, media_type="text/event-stream")
        try:
            return await responses_service.collect_response(response_request, run)
        except UnsupportedResponsesOutputError as exc:
            raise OpenAIHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error=ErrorObject(message=str(exc), type="server_error"),
            ) from exc


__all__ = ["router"]
