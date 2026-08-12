"""Dependencies for model routes."""

from fastapi import Request

from langgraph_openai_serve.graph.graph_registry import GraphRegistry


def get_graph_registry_dependency(request: Request) -> GraphRegistry:
    """Get the graph registry from application state."""
    return request.app.state.graph_registry
