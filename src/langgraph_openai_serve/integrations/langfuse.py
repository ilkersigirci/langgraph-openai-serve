"""Lazy construction for the optional Langfuse tracing integration."""

from contextlib import AbstractContextManager
from functools import cache
from typing import TYPE_CHECKING

from langchain_core.callbacks import BaseCallbackHandler

from langgraph_openai_serve.graph.telemetry import INSTRUMENTATION_SCOPE

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan


def should_export_span(span: "ReadableSpan") -> bool:
    """
    Apply Langfuse's default export filter to every span except those of LGOS.

    Langfuse exports spans with ``gen_ai.*`` attributes by default. LGOS's
    workflow span would then become each Langfuse trace's root observation in
    place of the callback's, which carries the run's input and output.
    """
    from langfuse.span_filter import is_default_export_span

    scope = span.instrumentation_scope
    if scope is not None and scope.name == INSTRUMENTATION_SCOPE:
        return False
    return is_default_export_span(span)


@cache
def get_langfuse_callback() -> BaseCallbackHandler:
    """Return the process-wide Langfuse callback, constructing it lazily."""
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # The first client of a Langfuse project configures its export, so an
    # application that already created one keeps its own filter.
    Langfuse(should_export_span=should_export_span)
    return CallbackHandler()


def trace_attributes(
    *, user_id: str | None, session_id: str | None
) -> AbstractContextManager[object]:
    """Return Langfuse's context naming the end user and session of observations."""
    from langfuse import propagate_attributes

    return propagate_attributes(user_id=user_id, session_id=session_id)
