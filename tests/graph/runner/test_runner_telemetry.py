from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing

import pytest
from anyio import Event, create_task_group, fail_after, sleep_forever
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from langgraph_openai_serve.core.errors import GraphError
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream
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
    # The measurement links to its run's trace.
    (exemplar,) = duration.exemplars
    assert exemplar.span_id == span.context.span_id


async def test_workflow_is_reported_when_its_final_output_arrives(
    make_request, telemetry: Telemetry
) -> None:
    events = run_langgraph_stream(
        make_request("workflow"),
        [HumanMessage(content="hi")],
        registry(make_message_graph()),
    )
    async with aclosing(events):
        async for event in events:
            if isinstance(event, AIMessage):
                assert len(telemetry.workflow_spans()) == 1
                assert len(telemetry.workflow_durations()) == 1


async def test_workflow_span_parents_graph_work_but_not_the_stream_consumer(
    make_request, telemetry: Telemetry
) -> None:
    tracer = trace.get_tracer(__name__)
    model = FakeListChatModel(responses=["hello world"])

    async def generate(state: MessageState) -> dict:
        with tracer.start_as_current_span("generate"):
            return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessageState).add_node("generate", generate)
    graph = graph.set_entry_point("generate").set_finish_point("generate").compile()

    with tracer.start_as_current_span("consumer") as consumer:
        events = run_langgraph_stream(
            make_request("workflow"), [HumanMessage(content="hi")], registry(graph)
        )
        async with aclosing(events):
            async for _event in events:
                assert trace.get_current_span() is consumer

    (workflow,) = telemetry.workflow_spans()
    assert workflow.parent == consumer.get_span_context()
    spans = {span.name: span for span in telemetry.span_exporter.get_finished_spans()}
    assert spans["generate"].parent == workflow.context


async def fail(_state: MessageState) -> dict:
    msg = "graph failed"
    raise ValueError(msg)


def interrupt_undeclared(_state: MessageState) -> dict:
    interrupt({"question": "Approve?"})
    return {}


@pytest.mark.parametrize(
    ("node", "error", "error_type"),
    [
        pytest.param(fail, ValueError, "ValueError", id="graph-exception"),
        # LGOS rejects the interrupt after the graph stream has ended.
        pytest.param(
            interrupt_undeclared,
            GraphError,
            "langgraph_openai_serve.core.errors.GraphError",
            id="output-error",
        ),
    ],
)
async def test_failed_run_reports_its_error_type(
    node, error: type[Exception], error_type: str, make_request, telemetry: Telemetry
) -> None:
    graph = StateGraph(MessageState).add_node("node", node)
    graph = graph.set_entry_point("node").set_finish_point("node").compile()

    with pytest.raises(error):
        await run_langgraph(
            make_request("workflow"), [HumanMessage(content="hi")], registry(graph)
        )

    (span,) = telemetry.workflow_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == error_type
    # Callers log the failure; an exception is recorded once.
    assert span.events == ()
    (duration,) = telemetry.workflow_durations()
    assert dict(duration.attributes or {}) == {
        "gen_ai.workflow.name": "workflow",
        "error.type": error_type,
    }


async def cancel_stream(events: AsyncGenerator[object, None]) -> None:
    started = Event()

    async def consume() -> None:
        async with aclosing(events):
            async for _event in events:
                started.set()

    async with create_task_group() as tasks:
        tasks.start_soon(consume)
        await started.wait()
        tasks.cancel_scope.cancel()


async def close_stream(events: AsyncGenerator[object, None]) -> None:
    async with aclosing(events):
        await anext(events)


@pytest.mark.parametrize(
    ("stop", "error_type"),
    [
        pytest.param(
            cancel_stream, "asyncio.exceptions.CancelledError", id="cancelled"
        ),
        pytest.param(close_stream, "GeneratorExit", id="stream-closed"),
    ],
)
async def test_run_stopped_before_its_output_is_a_failure(
    stop: Callable[[AsyncGenerator[object, None]], Awaitable[None]],
    error_type: str,
    make_request,
    telemetry: Telemetry,
) -> None:
    async def wait(_state: MessageState) -> dict:
        get_stream_writer()({"status": "working"})
        await sleep_forever()
        return {}

    graph = StateGraph(MessageState).add_node("wait", wait)
    graph = graph.set_entry_point("wait").set_finish_point("wait").compile()

    with fail_after(1):
        await stop(
            run_langgraph_stream(
                make_request("workflow"), [HumanMessage(content="hi")], registry(graph)
            )
        )

    (span,) = telemetry.workflow_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == error_type
    (duration,) = telemetry.workflow_durations()
    assert dict(duration.attributes or {}) == {
        "gen_ai.workflow.name": "workflow",
        "error.type": error_type,
    }
