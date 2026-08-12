"""Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Annotated

from anyio import CancelScope, create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.service import ChatCompletionService
from langgraph_openai_serve.api.chat.utils.interrupts import (
    InvalidInterruptPayloadError,
    InvalidResumeRequestError,
)
from langgraph_openai_serve.api.models.views import get_graph_registry_dependency
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from langgraph_openai_serve.graph.client_settings import ClientSettingsValidationError
from langgraph_openai_serve.graph.coordination import RunBusyError
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfigurationError,
    GraphNotFoundError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.utils import (
    RUN_METADATA_KEY,
    GraphRun,
    InterruptStateConflictError,
    InvalidRunIDError,
    prepare_run,
)
from langgraph_openai_serve.utils.message import InvalidChatMessageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])
CLIENT_ERROR_TYPES = (
    InvalidRunIDError,
    InvalidResumeRequestError,
    GraphNotFoundError,
    ClientSettingsValidationError,
    InvalidChatMessageError,
)


class _StreamOwner:
    """Own one streaming request's producer and source iterator."""

    def __init__(self) -> None:
        self._started = False
        self._producer: asyncio.Task[None] | None = None
        self._run: GraphRun | None = None
        self._send_stream: MemoryObjectSendStream[str] | None = None
        self._receive_stream: MemoryObjectReceiveStream[str] | None = None

    def start(
        self,
        source: AsyncGenerator[str, None],
        run: GraphRun,
    ) -> MemoryObjectReceiveStream[str]:
        if self._started:
            raise RuntimeError("A stream owner can only start one producer.")

        # An unbuffered handoff propagates response backpressure into graph
        # execution.
        send_stream, receive_stream = create_memory_object_stream[str](
            max_buffer_size=0
        )

        async def produce() -> None:
            async with aclosing(source), send_stream:
                async for chunk in source:
                    await send_stream.send(chunk)

        self._started = True
        self._run = run
        self._send_stream = send_stream
        self._receive_stream = receive_stream
        # The request dependency owns and joins this task outside Starlette's
        # response cancellation scope. Keep it asyncio-native: LangGraph stream
        # unwinding relies on edge cancellation, unlike AnyIO task groups.
        self._producer = asyncio.create_task(produce(), name="chat-completion-stream")
        return receive_stream

    async def aclose(self) -> None:
        producer = self._producer
        run = self._run
        if run is None:
            return

        # Cleanup may run inside the request's cancelled scope, so shield nested
        # stream finalizers long enough to finish.
        with CancelScope(shield=True):
            primary_error: BaseException | None = None
            try:
                await self._stop_producer(producer)
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                try:
                    await self._close_run(run, primary_error)
                finally:
                    self._reset()

    @staticmethod
    async def _stop_producer(
        producer: asyncio.Task[None] | None,
    ) -> None:
        if producer is None:
            return

        cancel_requested = False
        if not producer.done():
            cancel_requested = producer.cancel()

        try:
            await producer
        except asyncio.CancelledError:
            if not cancel_requested:
                raise

    @staticmethod
    async def _close_run(
        run: GraphRun,
        primary_error: BaseException | None,
    ) -> None:
        try:
            await run.aclose()
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Could not release a streaming graph-run lease during error cleanup."
            )

    def _reset(self) -> None:
        if self._send_stream is not None:
            self._send_stream.close()
        if self._receive_stream is not None:
            self._receive_stream.close()
        self._producer = None
        self._run = None
        self._send_stream = None
        self._receive_stream = None


async def _stream_owner_dependency() -> AsyncIterator[_StreamOwner]:
    owner = _StreamOwner()
    try:
        yield owner
    finally:
        await owner.aclose()


async def _checkpoint_scope_dependency(request: Request) -> str:
    value = request.app.state.checkpoint_scope(request)
    if inspect.isawaitable(value):
        value = await value
    return value


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    service: Annotated[ChatCompletionService, Depends(ChatCompletionService)],
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
    checkpoint_scope: Annotated[str, Depends(_checkpoint_scope_dependency)],
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
            checkpoint_scope=checkpoint_scope,
        )

        if chat_request.stream:
            logger.info("Streaming chat completion response")
            body = stream_owner.start(
                service.stream_completion(chat_request, run),
                run,
            )
            return StreamingResponse(
                body,
                media_type="text/event-stream",
            )

        logger.info("Generating non-streaming chat completion response")
        response = await service.generate_completion(chat_request, run)
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
    except CLIENT_ERROR_TYPES as e:
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


def client_error_param(error: Exception) -> str | None:
    if isinstance(error, GraphNotFoundError):
        return "model"
    if isinstance(error, InvalidRunIDError):
        return f"metadata.{RUN_METADATA_KEY}"
    if isinstance(error, InvalidResumeRequestError):
        return "messages"
    if isinstance(error, ClientSettingsValidationError):
        return error.param
    if isinstance(error, InvalidChatMessageError):
        return "messages"
    return None
