"""Lazy construction for the optional Langfuse tracing integration."""

from functools import cache

from langchain_core.callbacks import BaseCallbackHandler


@cache
def get_langfuse_callback() -> BaseCallbackHandler:
    """Return the process-wide Langfuse callback, constructing it lazily."""
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()
