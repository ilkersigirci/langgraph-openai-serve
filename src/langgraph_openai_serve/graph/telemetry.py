"""
Report graph runs with the OpenTelemetry GenAI workflow conventions.

LGOS uses only the OpenTelemetry API, so nothing is recorded until the
application configures an SDK.
"""

from collections.abc import AsyncGenerator, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter
from typing import TypeVar

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, StatusCode

from langgraph_openai_serve.core.logging import exception_type_name
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.protocol import CONVERSATION_METADATA_KEY

INSTRUMENTATION_SCOPE = "langgraph_openai_serve"
# The histogram boundaries the conventions specify for workflow durations.
_WORKFLOW_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200)

_tracer = trace.get_tracer(INSTRUMENTATION_SCOPE, version(INSTRUMENTATION_SCOPE))
_meter = metrics.get_meter(INSTRUMENTATION_SCOPE, version(INSTRUMENTATION_SCOPE))
_duration = _meter.create_histogram(
    "gen_ai.invoke_workflow.duration",
    unit="s",
    description="Duration of GenAI workflow.",
    explicit_bucket_boundaries_advisory=_WORKFLOW_BUCKETS,
)

_Event = TypeVar("_Event")


@dataclass(frozen=True)
class Workflow:
    """The span of one graph execution, current only while LGOS advances it."""

    span: Span

    def active(self) -> AbstractContextManager[Span]:
        """Make the span current; ``invoke_workflow`` alone sets its status."""
        return trace.use_span(
            self.span, record_exception=False, set_status_on_exception=False
        )

    async def iterate(
        self, events: AsyncGenerator[_Event, None]
    ) -> AsyncGenerator[_Event, None]:
        """
        Advance ``events`` with the span current, but yield them without it.

        A span current across ``yield`` would parent the consumer's work between
        events, and a stream closed from another context, such as by garbage
        collection, could not detach it.

        Yields:
            Each event of ``events``.

        """
        try:
            while True:
                with self.active():
                    try:
                        event = await anext(events)
                    except StopAsyncIteration:
                        return
                yield event
        finally:
            # Closing a stream early still runs graph work, such as LangGraph's
            # exit checkpoint write.
            with self.active():
                await events.aclose()


@contextmanager
def invoke_workflow(request: GraphRequest) -> Iterator[Workflow]:
    """
    Record one graph execution as an ``invoke_workflow`` span and duration.

    A run that ends before its output exists fails with its ``error.type``:
    an exception, a cancellation from a disconnecting client or a cancel
    request, or a consumer closing the stream mid-run.

    Yields:
        The run's workflow, whose span becomes current only through its methods.

    """
    attributes = {
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.workflow.name": request.model,
    }
    if conversation_id := request.metadata.get(CONVERSATION_METADATA_KEY):
        attributes["gen_ai.conversation.id"] = conversation_id
    measurement = {"gen_ai.workflow.name": request.model}
    span = _tracer.start_span(f"invoke_workflow {request.model}", attributes=attributes)
    started = perf_counter()
    try:
        yield Workflow(span)
    except BaseException as exc:
        # An exception should be recorded once, as a log where it is handled.
        # LGOS routes, background engines, and direct callers handle it, so the
        # span adds no exception event.
        error_type = exception_type_name(exc)
        span.set_status(StatusCode.ERROR, str(exc) or None)
        span.set_attribute("error.type", error_type)
        measurement["error.type"] = error_type
        raise
    finally:
        # The span is not current here, so pass it for exemplars to link the
        # measurement to this run's trace.
        _duration.record(
            perf_counter() - started,
            measurement,
            context=trace.set_span_in_context(span),
        )
        span.end()


__all__ = ["INSTRUMENTATION_SCOPE", "Workflow", "invoke_workflow"]
