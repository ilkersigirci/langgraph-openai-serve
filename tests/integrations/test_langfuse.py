from collections.abc import Iterator

import pytest
from langchain_core.messages import HumanMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from tests.graph.support.message import make_message_graph
from tests.graph.support.request import graph_request
from tests.graph.support.telemetry import Telemetry

from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.core.settings import Settings
from langgraph_openai_serve.graph.runner import run_langgraph
from langgraph_openai_serve.integrations.langfuse import get_langfuse_callback

langfuse = pytest.importorskip(
    "langfuse", reason="Langfuse is the optional tracing extra."
)


@pytest.fixture
def langfuse_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-lgos-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-lgos-test")
    # A closed local port, so anything Langfuse sends fails at once, and the
    # shortest timeout, so its exporter soon stops retrying at shutdown.
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGFUSE_TIMEOUT", "1")
    # Graph runs read the setting where they are prepared.
    monkeypatch.setattr(
        "langgraph_openai_serve.graph.run.settings", Settings(ENABLE_LANGFUSE=True)
    )
    get_langfuse_callback.cache_clear()
    yield
    get_langfuse_callback.cache_clear()
    # Langfuse keeps its clients for the life of the process, even after
    # shutdown, so a later client would reuse this one's configuration.
    from langfuse._client.resource_manager import LangfuseResourceManager  # ruff: ignore[import-private-name] - Langfuse has no public reset.

    LangfuseResourceManager.reset()


@pytest.fixture
def langfuse_spans(langfuse_enabled: None) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The first client of a Langfuse project configures its tracing, so the
    # client LGOS creates for the same project records into this provider.
    # Langfuse's processor still names the user and session as spans start,
    # but sends nothing to Langfuse.
    langfuse.Langfuse(tracer_provider=provider, should_export_span=lambda _span: False)
    return exporter


def messages_registry() -> GraphRegistry:
    return GraphRegistry(
        graphs={
            "messages": GraphConfig(graph=make_message_graph(), description="DUMMY")
        }
    )


async def test_langfuse_observations_stay_out_of_the_application_traces(
    langfuse_enabled: None,
    telemetry: Telemetry,
) -> None:
    request = graph_request(
        "messages",
        user="user-123",
        metadata={"conversation_id": "conversation-123"},
    )

    await run_langgraph(
        request, [HumanMessage(content="question")], messages_registry()
    )

    # Langfuse observations carry prompts and outputs, so only the workflow span
    # reaches the application's exporters, with only its GenAI attributes.
    (span,) = telemetry.span_exporter.get_finished_spans()
    assert span.name == "invoke_workflow messages"
    assert dict(span.attributes or {}) == {
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.workflow.name": "messages",
        "gen_ai.conversation.id": "conversation-123",
    }


async def test_every_langfuse_observation_names_its_user_session_and_graph(
    langfuse_spans: InMemorySpanExporter,
) -> None:
    request = graph_request(
        "messages",
        user="user-123",
        metadata={"conversation_id": "conversation-123"},
    )

    await run_langgraph(
        request, [HumanMessage(content="question")], messages_registry()
    )

    observations = langfuse_spans.get_finished_spans()
    # The root run and its node, whose model call carries the run's cost.
    assert {"lgos.graph_run", "generate"} <= {span.name for span in observations}
    assert {
        (
            span.name,
            (span.attributes or {}).get("user.id"),
            (span.attributes or {}).get("session.id"),
            (span.attributes or {}).get("langfuse.trace.name"),
        )
        for span in observations
    } == {
        (span.name, "user-123", "conversation-123", "messages") for span in observations
    }
