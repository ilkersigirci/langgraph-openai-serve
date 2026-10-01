"""
Report graph runs with the OpenTelemetry GenAI workflow conventions.

LGOS uses only the OpenTelemetry API, so nothing is recorded until the
application configures an SDK.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from time import perf_counter

from opentelemetry import metrics, trace

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
def invoke_workflow(request: GraphRequest) -> Iterator[None]:
    """
    Record one graph execution as an ``invoke_workflow`` span and duration.

    An exception marks the run failed with its ``error.type``. A cancellation,
    from a disconnecting client or a cancel request, is not a failure: it
    derives from ``BaseException``, which the API does not record as an error.
    """
    attributes = {
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.workflow.name": request.model,
    }
    if conversation_id := request.metadata.get(CONVERSATION_METADATA_KEY):
        attributes["gen_ai.conversation.id"] = conversation_id
    measurement = {"gen_ai.workflow.name": request.model}
    started = perf_counter()
    # An exception should be recorded once, as a log where it is handled. LGOS
    # routes, background engines, and direct callers handle it, so the span adds
    # no exception event.
    with _tracer.start_as_current_span(
        f"invoke_workflow {request.model}",
        attributes=attributes,
        record_exception=False,
    ) as span:
        try:
            yield
        except Exception as exc:
            error_type = exception_type_name(exc)
            span.set_attribute("error.type", error_type)
            measurement["error.type"] = error_type
            raise
        finally:
            _duration.record(perf_counter() - started, measurement)


__all__ = ["INSTRUMENTATION_SCOPE", "invoke_workflow"]
