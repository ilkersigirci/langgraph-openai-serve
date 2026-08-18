"""
Tie LangGraph stream production to a FastAPI request's lifetime.

Starlette owns response consumption, not the nested graph producer, so a client
disconnect may leave graph and provider work running. The request dependency
creates a ``_StreamOwner``; the route passes ``start()``'s receive stream to
``StreamingResponse``, and dependency cleanup cancels the producer and releases
its ``GraphRun``.

AnyIO still provides the channel and cleanup shield, but its task-group level
cancellation can repeatedly interrupt LangGraph's asyncio-native teardown. The
producer therefore remains an ``asyncio.Task`` so cancellation is delivered once
at the stream boundary.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing

from anyio import CancelScope, create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from langgraph_openai_serve.graph.utils import GraphRun

logger = logging.getLogger(__name__)


class _StreamOwner:
    """Own the producer and resources for one streaming graph run."""

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
            msg = "A stream owner can only start one producer."
            raise RuntimeError(msg)

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
