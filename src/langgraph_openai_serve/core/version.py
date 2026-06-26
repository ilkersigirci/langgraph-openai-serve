from functools import lru_cache
from importlib.metadata import version as metadata_version


@lru_cache
def get_version() -> str:
    """Return installed package version."""
    return metadata_version("langgraph_openai_serve")
