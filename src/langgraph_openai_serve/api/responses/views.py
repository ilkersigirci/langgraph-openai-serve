"""OpenAI-compatible Responses router."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
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
    validate_tools,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import UnsupportedResponsesOutputError
from langgraph_openai_serve.api.responses.streaming import (
    collect_response,
    stream_response,
)
from langgraph_openai_serve.api.streaming import _StreamOwner
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.core.logging import bind_log_context
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt.models import InterruptResume
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.graph.utils import prepare_run

router = APIRouter(tags=["openai"])


def _validate_responses_request(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
) -> tuple[GraphRequest, list[BaseMessage], InterruptResume | None]:
    graph_config = graph_registry.get_graph(request.model)
    validate_tools(request, graph_config.server_tools)
    if request.previous_response_id is not None and not graph_config.supports(
        GraphFeature.INTERRUPTS
    ):
        message = (
            "Previous response state is not supported for model "
            f"'{request.model}'; only interruptible graphs support "
            "'previous_response_id'."
        )
        raise UnsupportedResponsesRequestError(message, param="previous_response_id")
    return decode_responses_request(request, graph_config.server_tools)


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
            graph_request, messages, resume = _validate_responses_request(
                response_request,
                graph_registry,
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
            return await collect_response(response_request, run)
        except UnsupportedResponsesOutputError as exc:
            raise OpenAIHTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error=ErrorObject(message=str(exc), type="server_error"),
            ) from exc


__all__ = ["router"]
