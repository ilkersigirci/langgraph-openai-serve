from collections.abc import Iterator

import pytest
from langchain_core.messages import HumanMessage
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from tests.graph.support.message import make_message_graph
from tests.graph.support.request import graph_request

from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.core.settings import Settings
from langgraph_openai_serve.graph.runner import run_langgraph
from langgraph_openai_serve.integrations.langfuse import (
    get_langfuse_callback,
    should_export_span,
)

langfuse = pytest.importorskip(
    "langfuse", reason="Langfuse is the optional tracing extra."
)


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


@pytest.fixture
def langfuse_spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-lgos-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-lgos-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.invalid")
    # Graph runs read the setting where they are prepared.
    monkeypatch.setattr(
        "langgraph_openai_serve.graph.run.settings", Settings(ENABLE_LANGFUSE=True)
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The first client of a Langfuse project configures its tracing, so the
    # client LGOS creates for the same project records into this provider.
    # Langfuse's processor still names the user and session as spans start,
    # but sends nothing to Langfuse.
    langfuse.Langfuse(tracer_provider=provider, should_export_span=lambda _span: False)
    get_langfuse_callback.cache_clear()
    yield exporter
    get_langfuse_callback.cache_clear()
    # Langfuse keeps its clients for the life of the process, even after
    # shutdown, so a later client would reuse this one's provider.
    from langfuse._client.resource_manager import LangfuseResourceManager  # ruff: ignore[import-private-name] - Langfuse has no public reset.

    LangfuseResourceManager.reset()


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


async def test_every_langfuse_observation_of_a_run_names_its_user_and_session(
    langfuse_spans: InMemorySpanExporter,
) -> None:
    graph_registry = GraphRegistry(
        graphs={
            "messages": GraphConfig(graph=make_message_graph(), description="DUMMY")
        }
    )
    request = graph_request(
        "messages",
        user="user-123",
        metadata={"conversation_id": "conversation-123"},
    )

    await run_langgraph(request, [HumanMessage(content="question")], graph_registry)

    observations = langfuse_spans.get_finished_spans()
    # The root run and its node, whose model call carries the run's cost.
    assert {"lgos.graph_run", "generate"} <= {span.name for span in observations}
    assert {
        (
            span.name,
            (span.attributes or {}).get("user.id"),
            (span.attributes or {}).get("session.id"),
        )
        for span in observations
    } == {(span.name, "user-123", "conversation-123") for span in observations}
