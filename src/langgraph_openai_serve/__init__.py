"""langgraph-openai-serve package."""

from importlib.metadata import version

from langgraph_openai_serve.graph.graph_registry import (
    GraphCapabilities,
    GraphConfig,
    GraphRegistry,
)
from langgraph_openai_serve.openai_server import LangchainOpenaiApiServe

# Fetches the version of the package as defined in pyproject.toml
__version__ = version("langgraph_openai_serve")

__all__ = [
    "GraphCapabilities",
    "GraphConfig",
    "GraphRegistry",
    "LangchainOpenaiApiServe",
]
