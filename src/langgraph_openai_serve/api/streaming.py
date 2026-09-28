"""
Tie OpenAI stream production to a FastAPI request's lifetime.

Starlette owns response consumption, not the nested graph producer, so a client
disconnect may otherwise leave graph and provider work running. The producer is
a separate ``asyncio.Task`` because Starlette cancels the response through an
AnyIO scope, which cancels again at every await; LangGraph's asyncio-native
teardown must instead receive one cancellation at the stream boundary.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress

from anyio import CancelScope, create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from langgraph_openai_serve.graph.run import GraphRun


class StreamOwner:
    """Own the producer task and prepared run of one streaming request."""

    def __init__(self) -> None:
        self._producer: asyncio.Task[None] | None = None
        self._run: GraphRun | None = None
        self._streams: list[
            MemoryObjectSendStream[str] | MemoryObjectReceiveStream[str]
        ] = []

    def start(
        self,
        source: AsyncGenerator[str, None],
        run: GraphRun,
    ) -> MemoryObjectReceiveStream[str]:
        """Start producing ``source`` and close ``run`` when the request ends."""
        # An unbuffered handoff propagates response backpressure into graph
        # execution.
        send_stream, receive_stream = create_memory_object_stream[str](
            max_buffer_size=0
        )

        async def produce() -> None:
            async with send_stream, aclosing(source):
                async for chunk in source:
                    await send_stream.send(chunk)

        self._run = run
        self._streams = [send_stream, receive_stream]
        self._producer = asyncio.create_task(produce(), name="openai-response-stream")
        return receive_stream

    async def aclose(self) -> None:
        """Stop the producer, then close the run it may not have reached."""
        with CancelScope(shield=True):
            try:
                if self._producer is not None:
                    self._producer.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._producer
            finally:
                for stream in self._streams:
                    stream.close()
                if self._run is not None:
                    await self._run.aclose()


__all__ = ["StreamOwner"]
