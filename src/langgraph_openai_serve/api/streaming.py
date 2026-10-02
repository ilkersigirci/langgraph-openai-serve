"""
Tie OpenAI stream production to a FastAPI request's lifetime.

Starlette owns response consumption, not the nested graph producer, so a client
disconnect may otherwise leave graph and provider work running. The producer
runs as a ``TaskStream``, which request cleanup stops whether or not Starlette
is still consuming it.
"""

from collections.abc import AsyncGenerator

from anyio import CancelScope
from anyio.streams.memory import MemoryObjectReceiveStream

from langgraph_openai_serve.graph.run import GraphRun
from langgraph_openai_serve.graph.runner import TaskStream


class StreamOwner:
    """Own the producer task and prepared run of one streaming request."""

    def __init__(self) -> None:
        self._stream: TaskStream[str] | None = None
        self._run: GraphRun | None = None

    def start(
        self,
        source: AsyncGenerator[str, None],
        run: GraphRun,
    ) -> MemoryObjectReceiveStream[str]:
        """Start producing ``source`` and close ``run`` when the request ends."""
        self._run = run
        self._stream = TaskStream(source, name="openai-response-stream")
        return self._stream.receive

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
