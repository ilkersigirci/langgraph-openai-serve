"""Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import aclosing
from typing import Annotated, Protocol

from anyio import CancelScope, create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
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
from langgraph_openai_serve.graph.client_settings import ClientSettingsValidationError
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
    ClientSettingsValidationError,
    InvalidChatMessageError,
)
_StreamChunk = str | bytes | memoryview


class _ClosableAsyncIterable(Protocol):
    def __aiter__(self) -> AsyncIterator[_StreamChunk]: ...

    def aclose(self) -> Awaitable[None]: ...


class _StreamOwner:
    """Own one streaming request's producer and source iterator."""

    def __init__(self) -> None:
        self._started = False
        self._producer: asyncio.Task[None] | None = None
        self._source: _ClosableAsyncIterable | None = None
        self._send_stream: MemoryObjectSendStream[_StreamChunk] | None = None
        self._receive_stream: MemoryObjectReceiveStream[_StreamChunk] | None = None

    def start(
        self, source: _ClosableAsyncIterable
    ) -> MemoryObjectReceiveStream[_StreamChunk]:
        if self._started:
            raise RuntimeError("A stream owner can only start one producer.")

        send_stream, receive_stream = create_memory_object_stream[_StreamChunk](
            max_buffer_size=0
        )

        async def produce() -> None:
            async with send_stream:
                async for chunk in source:
                    await send_stream.send(chunk)

        self._started = True
        self._source = source
        self._send_stream = send_stream
        self._receive_stream = receive_stream
        # The request-scoped dependency owns this task outside Starlette's
        # response cancellation scope.
        self._producer = asyncio.create_task(produce(), name="chat-completion-stream")
        return receive_stream

    async def aclose(self) -> None:
        producer = self._producer
        source = self._source
        send_stream = self._send_stream
        receive_stream = self._receive_stream
        if producer is None or source is None:
            return

        cancel_requested = False
        if not producer.done():
            cancel_requested = producer.cancel()

        with CancelScope(shield=True):
            try:
                async with aclosing(source):
                    try:
                        await producer
                    except asyncio.CancelledError:
                        if not cancel_requested:
                            raise
            finally:
                if send_stream is not None:
                    send_stream.close()
                if receive_stream is not None:
                    receive_stream.close()
                self._producer = None
                self._source = None
                self._send_stream = None
                self._receive_stream = None


async def _stream_owner_dependency() -> AsyncIterator[_StreamOwner]:
    owner = _StreamOwner()
    try:
        yield owner
    finally:
        await owner.aclose()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    service: Annotated[ChatCompletionService, Depends(ChatCompletionService)],
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
    stream_owner: Annotated[
        _StreamOwner,
        Depends(_stream_owner_dependency, scope="request"),
    ],
) -> StreamingResponse | ChatCompletionResponse:
    """Create a chat completion.

    This endpoint is compatible with OpenAI's chat completion API.

    Args:
        chat_request: The parsed chat completion request.
        graph_registry: The graph registry dependency.
        service: The chat completion service dependency.
        stream_owner: The request-scoped streaming task owner.

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
            body = stream_owner.start(service.stream_completion(chat_request, run))
            return StreamingResponse(
                body,
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
    if isinstance(error, ClientSettingsValidationError):
        return error.param
    if isinstance(error, InvalidChatMessageError):
        return "messages"
    return None
