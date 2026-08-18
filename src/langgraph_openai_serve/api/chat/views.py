"""
Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.chat import service as chat_service
from langgraph_openai_serve.api.chat.deps import (
    checkpoint_scope_dependency,
    stream_owner_dependency,
)
from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.utils.interrupts import (
    InvalidInterruptPayloadError,
    InvalidResumeRequestError,
)
from langgraph_openai_serve.api.chat.utils.streaming import _StreamOwner
from langgraph_openai_serve.api.models.deps import get_graph_registry_dependency
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.client_settings import ClientSettingsValidationError
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfigurationError,
    GraphNotFoundError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.interrupt.coordination import RunBusyError
from langgraph_openai_serve.graph.interrupt.state import (
    RUN_METADATA_KEY,
    InterruptStateConflictError,
    InvalidRunIDError,
)
from langgraph_openai_serve.graph.utils import prepare_run
from langgraph_openai_serve.utils.message import InvalidChatMessageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])
_CLIENT_ERROR_TYPES = (
    InvalidRunIDError,
    InvalidResumeRequestError,
    GraphNotFoundError,
    ClientSettingsValidationError,
    InvalidChatMessageError,
)


def client_error_param(error: Exception) -> str | None:
    """Get client error param."""
    match error:
        case GraphNotFoundError():
            return "model"
        case InvalidRunIDError():
            return f"metadata.{RUN_METADATA_KEY}"
        case InvalidResumeRequestError() | InvalidChatMessageError():
            return "messages"
        case ClientSettingsValidationError():
            return error.param
        case _:
            return None


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
    logger.info(
        "Received chat completion request for model: %s, stream: %s",
        chat_request.model,
        chat_request.stream,
    )

    try:
        run = await prepare_run(
            chat_request.model,
            chat_request.messages,
            graph_registry,
            chat_request,
            checkpoint_scope=checkpoint_scope,
        )

        if chat_request.stream:
            logger.info("Streaming chat completion response")
            body = stream_owner.start(
                chat_service.stream_completion(chat_request, run),
                run,
            )
            return StreamingResponse(
                body,
                media_type="text/event-stream",
            )

        logger.info("Generating non-streaming chat completion response")
        response = await chat_service.generate_completion(chat_request, run)
    except RunBusyError as e:
        raise OpenAIHTTPException(
            status_code=status.HTTP_409_CONFLICT,
            error=ErrorObject(
                message=str(e),
                type="invalid_request_error",
                code="run_busy",
            ),
        ) from e
    except InterruptStateConflictError as e:
        raise OpenAIHTTPException(
            status_code=status.HTTP_409_CONFLICT,
            error=ErrorObject(
                message=str(e),
                type="invalid_request_error",
                param="messages",
                code="interrupt_state_conflict",
            ),
        ) from e
    except _CLIENT_ERROR_TYPES as e:
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message=str(e),
                type="invalid_request_error",
                param=client_error_param(e),
            ),
        ) from e
    except (GraphConfigurationError, InvalidInterruptPayloadError) as e:
        raise OpenAIHTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error=ErrorObject(
                message=str(e),
                type="server_error",
            ),
        ) from e
    logger.info("Returning non-streaming chat completion response")
    return response
