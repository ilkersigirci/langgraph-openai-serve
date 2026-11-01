from asyncio import CancelledError
from collections.abc import Awaitable, Callable
from contextlib import aclosing

import pytest
from anyio import Event, create_task_group, fail_after, sleep_forever
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.errors import NodeCancelledError
from langgraph.graph import END, START, StateGraph

from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream
from tests.graph.support.schemas import (
    AnswerOutput,
    MessageState,
    QuestionInput,
    QuestionState,
)


async def stream_text(name: str, graph_registry: GraphRegistry, make_request) -> str:
    request = make_request(name)
    chunks = run_langgraph_stream(
        request, [HumanMessage(content="question")], graph_registry
    )
    events = [event async for event in chunks]
    assert isinstance(events[-1], AIMessage)
    return "".join(event for event in events if isinstance(event, str))


async def test_nested_subgraph_streaming(
    make_request,
) -> None:
    model = FakeListChatModel(responses=["nested"])

    async def generate(state: QuestionState):
        await model.ainvoke([HumanMessage(content=state["question"])])
        return {"answer": "done"}

    subgraph = (
        StateGraph(QuestionState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    graph = (
        StateGraph(
            QuestionState,
            input_schema=QuestionInput,
            output_schema=AnswerOutput,
        )
        .add_node("subgraph", subgraph)
        .set_entry_point("subgraph")
        .set_finish_point("subgraph")
        .compile()
    )
    graph_registry = GraphRegistry(
        graphs={
            "nested": GraphConfig(
                graph=graph,
                description="DUMMY",
                request_to_input=lambda request, messages: {
                    "question": messages[-1].content
                },
                output_to_message=lambda output: AIMessage(content=output["answer"]),
            )
        },
    )

    assert await stream_text("nested", graph_registry, make_request) == "nested"


async def test_stream_excludes_disabled_model_streams_and_non_ai_messages(
    make_request,
) -> None:
    draft_model = FakeListChatModel(responses=["draft"])
    hidden_model = FakeListChatModel(
        responses=["hidden"],
        disable_streaming=True,
    )
    visible_model = FakeListChatModel(responses=["visible"])

    async def draft(state: MessageState):
        return {"messages": [await draft_model.ainvoke(state["messages"])]}

    async def generate(state: MessageState):
        await hidden_model.ainvoke(state["messages"])
        return {"messages": [await visible_model.ainvoke(state["messages"])]}

    builder = StateGraph(MessageState)
    builder.add_node(
        "non_ai",
        lambda state: {"messages": [HumanMessage(content="ignored")]},
    )
    builder.add_node("draft", draft)
    builder.add_node("generate", generate)
    builder.set_entry_point("non_ai")
    builder.add_edge("non_ai", "draft")
    builder.add_edge("draft", "generate")
    builder.set_finish_point("generate")

    graph_registry = GraphRegistry(
        graphs={
            "filtered": GraphConfig(
                graph=builder.compile(),
                description="DUMMY",
            )
        },
    )

    assert await stream_text("filtered", graph_registry, make_request) == "draftvisible"


async def test_stream_preserves_custom_events_with_their_namespace(
    make_request,
) -> None:
    payload = {"type": "progress", "data": {"completed": 2, "total": 5}}
    model = FakeListChatModel(responses=["ok"])

    async def research(state: MessageState):
        get_stream_writer()(payload)
        return {"messages": [await model.ainvoke(state["messages"])]}

    subgraph = (
        StateGraph(MessageState)
        .add_node("research", research)
        .set_entry_point("research")
        .set_finish_point("research")
        .compile()
    )
    graph = (
        StateGraph(MessageState)
        .add_node("subgraph", subgraph)
        .set_entry_point("subgraph")
        .set_finish_point("subgraph")
        .compile()
    )
    graph_registry = GraphRegistry(
        graphs={"events": GraphConfig(graph=graph, description="DUMMY")}
    )

    events = [
        event
        async for event in run_langgraph_stream(
            make_request("events"), [HumanMessage(content="hi")], graph_registry
        )
    ]

    custom, *text, final = events
    assert custom["type"] == "custom"
    assert custom["data"] == payload
    assert custom["ns"][0].startswith("subgraph:")
    assert "".join(text) == final.text == "ok"


async def stream_events(request, graph_registry: GraphRegistry) -> None:
    events = run_langgraph_stream(request, [HumanMessage(content="hi")], graph_registry)
    async with aclosing(events):
        async for _event in events:
            pass


async def invoke(request, graph_registry: GraphRegistry) -> None:
    await run_langgraph(request, [HumanMessage(content="hi")], graph_registry)


@pytest.mark.parametrize(
    ("run", "nodes"),
    [
        pytest.param(stream_events, 1, id="token-stream"),
        pytest.param(invoke, 2, id="parallel-nodes"),
    ],
)
async def test_anyio_cancellation_stops_graph_work(
    run: Callable[[object, GraphRegistry], Awaitable[None]],
    nodes: int,
    make_request,
) -> None:
    all_started, all_stopped = Event(), Event()
    started: list[None] = []
    stopped: list[None] = []

    async def wait(_state: MessageState) -> dict:
        # A node cancelled before it starts never reaches its finally block.
        started.append(None)
        if len(started) == nodes:
            all_started.set()
        try:
            await sleep_forever()
        finally:
            stopped.append(None)
            if len(stopped) == nodes:
                all_stopped.set()
        return {}

    graph = StateGraph(MessageState)
    for index in range(nodes):
        name = f"wait-{index}"
        graph = graph.add_node(name, wait).add_edge(START, name).add_edge(name, END)
    registry = GraphRegistry(
        graphs={"wait": GraphConfig(graph=graph.compile(), description="DUMMY")}
    )

    with fail_after(1):
        # A cancel scope cancels again at every await, unlike one asyncio cancel.
        async with create_task_group() as tasks:
            tasks.start_soon(run, make_request("wait"), registry)
            await all_started.wait()
            tasks.cancel_scope.cancel()
        await all_stopped.wait()


@pytest.mark.parametrize("run", [stream_events, invoke], ids=["streamed", "collected"])
async def test_node_that_cancels_itself_fails_the_run(
    run: Callable[[object, GraphRegistry], Awaitable[None]],
    make_request,
) -> None:
    async def draft(_state: MessageState) -> dict:
        return {"messages": [AIMessage(content="draft")]}

    async def review(_state: MessageState) -> dict:
        # As when a node awaits work that another task cancelled.
        raise CancelledError

    graph = (
        StateGraph(MessageState)
        .add_node("draft", draft)
        .add_node("review", review)
        .add_edge(START, "draft")
        .add_edge("draft", "review")
        .add_edge("review", END)
    )
    registry = GraphRegistry(
        graphs={"review": GraphConfig(graph=graph.compile(), description="DUMMY")}
    )

    # The run fails as an ordinary exception rather than answering with the draft.
    with pytest.raises(NodeCancelledError):
        await run(make_request("review"), registry)
