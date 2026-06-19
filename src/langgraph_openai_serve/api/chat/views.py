"""Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.service import ChatCompletionService
from langgraph_openai_serve.api.models.views import get_graph_registry_dependency
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])


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
        graph_config = graph_registry.get_graph(chat_request.model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    ui_event_options = chat_request.ui_event_options()
    if (
        ui_event_options.enabled
        and graph_config.capabilities.hitl
        and not ui_event_options.thread_id
    ):
        raise HTTPException(
            status_code=400,
            detail="UI-event HITL graphs require x_langgraph_openai_serve.thread_id",
        )

    if chat_request.stream:
        logger.info("Streaming chat completion response")
        return StreamingResponse(
            service.stream_completion(chat_request, graph_registry),
            media_type="text/event-stream",
        )

    logger.info("Generating non-streaming chat completion response")
    response = await service.generate_completion(chat_request, graph_registry)
    logger.info("Returning non-streaming chat completion response")
    return response
