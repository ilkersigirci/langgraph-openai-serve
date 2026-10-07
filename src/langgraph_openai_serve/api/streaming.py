"""
Tie OpenAI stream production to a FastAPI request's lifetime.

Starlette owns response consumption, not the nested graph producer, so a client
disconnect may otherwise leave graph and provider work running. The producer
runs as a ``TaskStream``, which request cleanup stops whether or not Starlette
is still consuming it.
"""

from collections.abc import AsyncGenerator, AsyncIterator

from anyio import CancelScope, EndOfStream, move_on_after
from anyio.streams.memory import MemoryObjectReceiveStream
from fastapi.responses import StreamingResponse

from langgraph_openai_serve.graph.run import GraphRun
from langgraph_openai_serve.graph.runner import TaskStream

# An SSE comment, which clients ignore, keeps idle connections open through
# proxies. The interval matches FastAPI's native SSE keepalive.
_KEEPALIVE = ": ping\n\n"
_KEEPALIVE_INTERVAL = 15.0
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # Nginx buffers proxied responses unless told otherwise.
    "X-Accel-Buffering": "no",
}


class StreamOwner:
    """Own the producer task and prepared run of one streaming request."""

    def __init__(self, keepalive_interval: float = _KEEPALIVE_INTERVAL) -> None:
        self._keepalive_interval = keepalive_interval
        self._stream: TaskStream[str] | None = None
        self._run: GraphRun | None = None

    def start(
        self,
        source: AsyncGenerator[str, None],
        run: GraphRun,
    ) -> StreamingResponse:
        """Stream ``source`` as SSE and close ``run`` when the request ends."""
        self._run = run
        self._stream = TaskStream(source, name="openai-response-stream")
        return StreamingResponse(
            self._with_keepalive(self._stream.receive),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    async def _with_keepalive(
        self, chunks: MemoryObjectReceiveStream[str]
    ) -> AsyncIterator[str]:
        while True:
            chunk = _KEEPALIVE
            # The timeout wraps only the receive, which keeps an item it times
            # out on, and never a yield to Starlette.
            with move_on_after(self._keepalive_interval):
                try:
                    chunk = await chunks.receive()
                except EndOfStream:
                    return
            yield chunk

    async def aclose(self) -> None:
        """Stop the producer, then close the run it may not have reached."""
        with CancelScope(shield=True):
            try:
                if self._stream is not None:
                    await self._stream.aclose()
            finally:
                if self._run is not None:
                    await self._run.aclose()


__all__ = ["StreamOwner"]
