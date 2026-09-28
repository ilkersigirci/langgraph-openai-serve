"""Dependencies shared by OpenAI-compatible API routes."""

from collections.abc import AsyncIterator

from fastapi import Request

from langgraph_openai_serve.api.streaming import StreamOwner
from langgraph_openai_serve.graph.graph_registry import GraphRegistry


def get_graph_registry(request: Request) -> GraphRegistry:
    """Get the graph registry from application state."""
    return request.app.state.graph_registry


async def get_stream_owner() -> AsyncIterator[StreamOwner]:
    """
    Provide one request-scoped stream owner.

    Yields:
        The stream owner, closed after the response finishes.

    """
    owner = StreamOwner()
    try:
        yield owner
    finally:
        await owner.aclose()


__all__ = ["get_graph_registry", "get_stream_owner"]
