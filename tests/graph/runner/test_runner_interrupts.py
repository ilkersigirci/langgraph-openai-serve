import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.runner import (
    LangGraphInterrupt,
    invoke_run,
    run_langgraph,
    run_langgraph_stream,
)
from langgraph_openai_serve.graph.utils import GraphRun
from tests.graph.support.interrupt import (
    DEFAULT_INTERRUPT_PAYLOAD,
    make_interrupt_graph,
)
from tests.graph.support.message import make_message_graph
from tests.graph.support.schemas import MessageState


@pytest.mark.anyio
async def test_thread_id_reaches_runnable_config(make_request) -> None:
    seen_thread_ids = []

    async def generate(state: MessageState, config: RunnableConfig):
        seen_thread_ids.append(config["configurable"]["thread_id"])
        return {"messages": [AIMessage(content="ok")]}

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile(checkpointer=InMemorySaver())
    )
    registry = GraphRegistry(
        registry={"threaded": GraphConfig(graph=graph, interrupts_enabled=True)}
    )
    request = make_request(
        "threaded",
        metadata={"langgraph_thread_id": "thread-1"},
    )

    response, _ = await run_langgraph("threaded", request.messages, registry, request)

    assert response == "ok"
    assert seen_thread_ids == ["thread-1"]


@pytest.mark.anyio
async def test_interrupt_result_is_returned_before_output_rendering(make_request) -> None:
    async def output_to_text(output):
        raise AssertionError("interrupt output should not be rendered")

    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(),
                output_to_text=output_to_text,
                interrupts_enabled=True,
            )
        }
    )
    request = make_request(
        "interruptible",
        metadata={"langgraph_thread_id": "thread-1"},
    )

    response, usage = await run_langgraph(
        "interruptible",
        request.messages,
        registry,
        request,
    )

    assert isinstance(response, LangGraphInterrupt)
    assert response.thread_id == "thread-1"
    assert response.payload == DEFAULT_INTERRUPT_PAYLOAD
    assert usage["completion_tokens"] == 0


@pytest.mark.anyio
async def test_interrupt_shape_is_ignored_when_interrupts_disabled() -> None:
    class Graph:
        async def ainvoke(self, inputs, *, config, context):
            return {"__interrupt__": ["not-enabled"]}

    async def output_to_text(output):
        return output["__interrupt__"][0]

    graph_config = GraphConfig(
        graph=make_interrupt_graph(),
        output_to_text=output_to_text,
    )
    run = GraphRun(
        config=graph_config,
        graph=Graph(),
        inputs={},
        context=None,
        runnable_config=None,
        thread_id=None,
    )

    response = await invoke_run(run)

    assert response == "not-enabled"


@pytest.mark.anyio
async def test_streaming_interrupt_detected_from_updates(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(),
                interrupts_enabled=True,
            )
        }
    )
    request = make_request(
        "interruptible",
        metadata={"langgraph_thread_id": "thread-1"},
    )

    events = [
        event
        async for event in run_langgraph_stream(
            "interruptible",
            request.messages,
            registry,
            request,
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], LangGraphInterrupt)
    assert events[0].payload == DEFAULT_INTERRUPT_PAYLOAD


@pytest.mark.anyio
async def test_interrupt_enabled_graph_requires_thread_id(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(),
                interrupts_enabled=True,
            )
        }
    )
    request = make_request("interruptible")

    with pytest.raises(ValueError, match=r"metadata\.langgraph_thread_id"):
        await run_langgraph("interruptible", request.messages, registry, request)


@pytest.mark.anyio
async def test_interrupt_enabled_graph_requires_checkpointer(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "broken": GraphConfig(
                graph=make_message_graph("ok"),
                interrupts_enabled=True,
            )
        }
    )
    request = make_request("broken", metadata={"langgraph_thread_id": "thread-1"})

    with pytest.raises(GraphConfigurationError, match="checkpointer"):
        await run_langgraph("broken", request.messages, registry, request)
