"""Lazy construction for the optional Langfuse tracing integration."""

from contextlib import AbstractContextManager
from functools import cache

from langchain_core.callbacks import BaseCallbackHandler


@cache
def get_langfuse_callback() -> BaseCallbackHandler:
    """Return the process-wide Langfuse callback, constructing it lazily."""
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    from opentelemetry.sdk.trace import TracerProvider

    # A tracer provider of its own keeps Langfuse's observations, which carry
    # prompts and outputs, out of the application's trace exporters, and the
    # application's spans out of Langfuse; both still share each trace ID. The
    # first client of a Langfuse project configures its tracing, so an
    # application that already created one keeps its own.
    Langfuse(tracer_provider=TracerProvider())
    return CallbackHandler()


def trace_attributes(
    *, user_id: str | None, session_id: str | None, trace_name: str
) -> AbstractContextManager[object]:
    """Return Langfuse's context naming the end user, session, and trace."""
    from langfuse import propagate_attributes

    return propagate_attributes(
        user_id=user_id, session_id=session_id, trace_name=trace_name
    )


__all__ = ["get_langfuse_callback"]
