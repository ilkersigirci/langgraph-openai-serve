import pytest
from anyio import Event, create_task_group, sleep_forever
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from opentelemetry.trace import SpanKind, StatusCode

from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph
from tests.graph.support.message import make_message_graph
from tests.graph.support.schemas import MessageState
from tests.graph.support.telemetry import Telemetry

# The boundaries the GenAI conventions specify for workflow durations.
CONVENTION_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200)


def registry(graph) -> GraphRegistry:
    return GraphRegistry(
        graphs={"workflow": GraphConfig(graph=graph, description="DUMMY")}
    )


async def test_graph_run_is_reported_as_a_genai_workflow(
    make_request, telemetry: Telemetry
) -> None:
    request = make_request("workflow", metadata={"conversation_id": "chat-1"})

    await run_langgraph(
        request, [HumanMessage(content="hi")], registry(make_message_graph())
    )

    (span,) = telemetry.workflow_spans()
    assert span.name == "invoke_workflow workflow"
    assert span.kind is SpanKind.INTERNAL
    assert span.status.status_code is StatusCode.UNSET
    assert dict(span.attributes or {}) == {
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.workflow.name": "workflow",
        "gen_ai.conversation.id": "chat-1",
    }
    (duration,) = telemetry.workflow_durations()
    assert dict(duration.attributes or {}) == {"gen_ai.workflow.name": "workflow"}
    assert duration.count == 1
    assert duration.explicit_bounds == CONVENTION_BUCKETS


async def test_failed_run_reports_its_error_type(
    make_request, telemetry: Telemetry
) -> None:
    async def fail(_state: MessageState) -> dict:
        msg = "graph failed"
        raise ValueError(msg)

    graph = StateGraph(MessageState).add_node("fail", fail)
    graph = graph.set_entry_point("fail").set_finish_point("fail").compile()

    with pytest.raises(ValueError, match="graph failed"):
        await run_langgraph(
            make_request("workflow"), [HumanMessage(content="hi")], registry(graph)
        )

    (span,) = telemetry.workflow_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "ValueError"
    # Callers log the failure; an exception is recorded once.
    assert span.events == ()
    (duration,) = telemetry.workflow_durations()
    assert dict(duration.attributes or {}) == {
        "gen_ai.workflow.name": "workflow",
        "error.type": "ValueError",
    }


async def test_cancelled_run_is_not_a_failure(
    make_request, telemetry: Telemetry
) -> None:
    started = Event()

    async def wait(_state: MessageState) -> dict:
        started.set()
        await sleep_forever()
        return {}

    graph = StateGraph(MessageState).add_node("wait", wait)
    graph = graph.set_entry_point("wait").set_finish_point("wait").compile()

    async with create_task_group() as tasks:
        tasks.start_soon(
            run_langgraph,
            make_request("workflow"),
            [HumanMessage(content="hi")],
            registry(graph),
        )
        await started.wait()
        tasks.cancel_scope.cancel()

    (span,) = telemetry.workflow_spans()
    assert span.status.status_code is StatusCode.UNSET
    assert "error.type" not in (span.attributes or {})
    (duration,) = telemetry.workflow_durations()
    assert dict(duration.attributes or {}) == {"gen_ai.workflow.name": "workflow"}
