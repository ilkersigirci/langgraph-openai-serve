"""Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.service import ChatCompletionService
from langgraph_openai_serve.api.chat.utils.interrupts import (
    InvalidResumeRequestError,
)
from langgraph_openai_serve.api.models.views import get_graph_registry_dependency
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfigurationError,
    GraphNotFoundError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.utils import (
    THREAD_METADATA_KEY,
    MissingThreadIDError,
    prepare_run,
)
from langgraph_openai_serve.utils.message import InvalidChatMessageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])
CLIENT_ERROR_TYPES = (
    MissingThreadIDError,
    InvalidResumeRequestError,
    GraphNotFoundError,
    InvalidChatMessageError,
)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    service: Annotated[ChatCompletionService, Depends(ChatCompletionService)],
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
) -> StreamingResponse | ChatCompletionResponse:
    """Create a chat completion.

    This endpoint is compatible with OpenAI's chat completion API.

    Args:
        chat_request: The parsed chat completion request.
        graph_registry: The graph registry dependency.
        service: The chat completion service dependency.

    Returns:
        A chat completion response, either as a complete response or as a stream.
    """

    logger.info(
        f"Received chat completion request for model: {chat_request.model}, "
        f"stream: {chat_request.stream}"
    )

    try:
        run = await prepare_run(
            chat_request.model,
            chat_request.messages,
            graph_registry,
            chat_request,
        )

        if chat_request.stream:
            logger.info("Streaming chat completion response")
            return StreamingResponse(
                service.stream_completion(chat_request, run),
                media_type="text/event-stream",
            )

        logger.info("Generating non-streaming chat completion response")
        response = await service.generate_completion(chat_request, run)
    except CLIENT_ERROR_TYPES as e:
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message=str(e),
                type="invalid_request_error",
                param=client_error_param(e),
            ),
        ) from e
    except GraphConfigurationError as e:
        raise OpenAIHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error=ErrorObject(
                message=str(e),
                type="server_error",
            ),
        ) from e
    logger.info("Returning non-streaming chat completion response")
    return response


def client_error_param(error: Exception) -> str | None:
    if isinstance(error, GraphNotFoundError):
        return "model"
    if isinstance(error, MissingThreadIDError):
        return f"metadata.{THREAD_METADATA_KEY}"
    if isinstance(error, InvalidResumeRequestError):
        return "messages"
    if isinstance(error, InvalidChatMessageError):
        return "messages"
    return None
