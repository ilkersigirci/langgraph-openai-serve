import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph

from langgraph_openai_serve.graph.features import GraphFeature
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

pytestmark = pytest.mark.anyio


async def test_thread_id_reaches_runnable_config(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    seen_thread_ids = []

    async def generate(state: MessageState, config: RunnableConfig):
        seen_thread_ids.append(config["configurable"]["thread_id"])
        return {"messages": [AIMessage(content="ok")]}

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile(checkpointer=sqlite_checkpointer)
    )
    registry = GraphRegistry(
        registry={
            "threaded": GraphConfig(
                graph=graph,
                features={GraphFeature.INTERRUPTS},
            )
        }
    )
    request = make_request(
        "threaded",
        metadata={"langgraph_thread_id": "thread-1"},
    )

    invocation = await run_langgraph("threaded", request.messages, registry, request)

    assert invocation.output == "ok"
    assert seen_thread_ids == ["thread-1"]


async def test_interrupt_result_is_returned_before_output_rendering(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    async def output_to_text(output):
        raise AssertionError("interrupt output should not be rendered")

    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
                output_to_text=output_to_text,
                features={GraphFeature.INTERRUPTS},
            )
        }
    )
    request = make_request(
        "interruptible",
        metadata={"langgraph_thread_id": "thread-1"},
    )

    invocation = await run_langgraph(
        "interruptible",
        request.messages,
        registry,
        request,
    )

    assert isinstance(invocation.output, LangGraphInterrupt)
    assert invocation.output.thread_id == "thread-1"
    assert invocation.output.payload == DEFAULT_INTERRUPT_PAYLOAD
    assert invocation.custom_events == ()


async def test_interrupt_shape_is_ignored_when_interrupts_disabled(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    class Graph:
        output_channels = ("__interrupt__",)

        async def astream(self, *args, **kwargs):
            yield {
                "type": "values",
                "ns": (),
                "data": {"__interrupt__": ["not-enabled"]},
            }

    async def output_to_text(output):
        return output["__interrupt__"][0]

    graph_config = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
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

    invocation = await invoke_run(run)

    assert invocation.output == "not-enabled"
    assert invocation.custom_events == ()


async def test_streaming_interrupt_detected_from_updates(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
                features={GraphFeature.INTERRUPTS},
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


async def test_interrupt_enabled_graph_requires_thread_id(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = GraphRegistry(
        registry={
            "interruptible": GraphConfig(
                graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
                features={GraphFeature.INTERRUPTS},
            )
        }
    )
    request = make_request("interruptible")

    with pytest.raises(ValueError, match=r"metadata\.langgraph_thread_id"):
        await run_langgraph("interruptible", request.messages, registry, request)


async def test_interrupt_enabled_graph_requires_checkpointer(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "broken": GraphConfig(
                graph=make_message_graph("ok"),
                features={GraphFeature.INTERRUPTS},
            )
        }
    )
    request = make_request("broken", metadata={"langgraph_thread_id": "thread-1"})

    with pytest.raises(GraphConfigurationError, match="checkpointer"):
        await run_langgraph("broken", request.messages, registry, request)
