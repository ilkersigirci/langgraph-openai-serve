"""langgraph-openai-serve package."""

from importlib.metadata import version

from langgraph_openai_serve.graph.client_settings import ClientSettings
from langgraph_openai_serve.graph.events import citation_event, citation_slice
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphRegistry,
)
from langgraph_openai_serve.openai_server import LanggraphOpenaiServe

# Fetches the version of the package as defined in pyproject.toml
__version__ = version("langgraph_openai_serve")

__all__ = [
    "ClientSettings",
    "GraphConfig",
    "GraphFeature",
    "GraphRegistry",
    "LanggraphOpenaiServe",
    "citation_event",
    "citation_slice",
]
