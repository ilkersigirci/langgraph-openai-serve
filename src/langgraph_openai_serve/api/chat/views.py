"""OpenAI-compatible Chat Completions router."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.chat import service as chat_service
from langgraph_openai_serve.api.chat.messages import InvalidChatMessageError
from langgraph_openai_serve.api.chat.request import decode_chat_request
from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.deps import (
    checkpoint_scope_dependency,
    stream_owner_dependency,
)
from langgraph_openai_serve.api.errors import graph_errors
from langgraph_openai_serve.api.models.deps import get_graph_registry_dependency
from langgraph_openai_serve.api.streaming import _StreamOwner
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.core.logging import bind_log_context
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.utils import prepare_run

router = APIRouter(tags=["openai"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
    checkpoint_scope: Annotated[str, Depends(checkpoint_scope_dependency)],
    stream_owner: Annotated[
        _StreamOwner,
        Depends(stream_owner_dependency, scope="request"),
    ],
) -> StreamingResponse | ChatCompletionResponse:
    """
    Create a chat completion.

    This endpoint is compatible with OpenAI's chat completion API.

    Args:
        chat_request: The parsed chat completion request.
        graph_registry: The graph registry dependency.
        checkpoint_scope: The checkpoint scope boundary.
        stream_owner: The request-scoped streaming task owner.

    Returns:
        A chat completion response, either as a complete response or as a stream.

    """
    bind_log_context(
        model=chat_request.model,
        stream=chat_request.stream,
    )

    with graph_errors(input_param="messages"):
        try:
            graph_request, messages, resume = decode_chat_request(chat_request)
        except InvalidChatMessageError as exc:
            raise OpenAIHTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                error=ErrorObject(
                    message=str(exc), type="invalid_request_error", param="messages"
                ),
            ) from exc
        run = await prepare_run(
            graph_request,
            messages,
            graph_registry,
            resume=resume,
            checkpoint_scope=checkpoint_scope,
        )

        if chat_request.stream:
            body = stream_owner.start(
                chat_service.stream_completion(chat_request, run),
                run,
            )
            return StreamingResponse(
                body,
                media_type="text/event-stream",
            )

        return await chat_service.generate_completion(chat_request, run)
