"""Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from langgraph_openai_serve.api.chat.service import ChatCompletionService
from langgraph_openai_serve.schemas.openai_schema import ChatCompletionRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])
chat_service = ChatCompletionService()


@router.post("/chat/completions")
async def create_chat_completion(
    request: Request, chat_request: ChatCompletionRequest
) -> JSONResponse:
    """Create a chat completion.

    This endpoint is compatible with OpenAI's chat completion API.

    Args:
        request: The incoming HTTP request.
        chat_request: The parsed chat completion request.

    Returns:
        A chat completion response, either as a complete response or as a stream.
    """

    logger.info(f"Received chat completion request for model: {chat_request.model}")

    if chat_request.stream is True:
        return StreamingResponse(
            chat_service.stream_completion(chat_request),
            media_type="text/event-stream",
        )

    try:
        response = await chat_service.generate_completion(chat_request)
        return JSONResponse(response.model_dump())

    except Exception as e:
        logger.exception("Error generating chat completion")
        raise HTTPException(status_code=500, detail=str(e)) from e
