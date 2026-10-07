"""
Report graph runs with the OpenTelemetry GenAI workflow conventions.

LGOS uses only the OpenTelemetry API, so nothing is recorded until the
application configures an SDK.
"""

from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import version
from time import perf_counter

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


@contextmanager
def invoke_workflow(request: GraphRequest) -> Generator[Span, None, None]:
    """
    Record one graph execution as an ``invoke_workflow`` span and duration.

    A run that ends before its output exists fails with its ``error.type``:
    an exception, a cancellation from a disconnecting client or a cancel
    request, or a consumer closing the stream mid-run.

    Yields:
        The run's span, which callers make current only around graph work.

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
        yield span
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


__all__ = ["INSTRUMENTATION_SCOPE", "invoke_workflow"]
