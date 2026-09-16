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
    Manage the streaming producer owned by one request.

    Yields:
        The request-scoped stream owner.

    """
    async with StreamOwner() as owner:
        yield owner


__all__ = ["get_graph_registry", "get_stream_owner"]
