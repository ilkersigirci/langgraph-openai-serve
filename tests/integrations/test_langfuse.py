import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from langgraph_openai_serve.integrations.langfuse import should_export_span

pytest.importorskip("langfuse", reason="Langfuse is the optional tracing extra.")


def finished_span(scope: str, attributes: dict[str, str]) -> ReadableSpan:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer(scope).start_as_current_span(
        "span", attributes=attributes
    ):
        pass
    (span,) = exporter.get_finished_spans()
    return span


@pytest.mark.parametrize(
    ("scope", "attributes", "exported"),
    [
        pytest.param(
            "langgraph_openai_serve",
            {"gen_ai.operation.name": "invoke_workflow"},
            False,
            id="lgos-workflow",
        ),
        pytest.param(
            "other.framework",
            {"gen_ai.operation.name": "invoke_agent"},
            True,
            id="other-genai",
        ),
        pytest.param(
            "other.framework", {"http.route": "/v1/responses"}, False, id="other-http"
        ),
    ],
)
def test_langfuse_exports_its_default_spans_except_lgos_spans(
    scope: str, attributes: dict[str, str], exported: bool
) -> None:
    assert should_export_span(finished_span(scope, attributes)) is exported
