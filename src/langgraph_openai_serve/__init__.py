"""langgraph-openai-serve package."""

from importlib.metadata import version

from langgraph_openai_serve.graph_serve import LangchainOpenaiApiServe

# Fetches the version of the package as defined in pyproject.toml
__version__ = version("langgraph_openai_serve")

__all__ = ["LangchainOpenaiApiServe"]
