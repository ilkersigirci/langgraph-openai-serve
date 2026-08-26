from typing import Any, cast

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import CustomStreamPart
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

from lgos_demo_api.graphs.persistent_plot import (
    ARTIFACT_KEY,
    PersistentPlotContext,
    PlotDocument,
    create_persistent_plot_graph,
    create_persistent_plot_graph_config,
)


def _public_event(value: object) -> dict[str, Any]:
    part = cast(CustomStreamPart, value)
    envelope = cast(dict[str, Any], part["data"])
    return cast(dict[str, Any], envelope["event"])


def _state(prompt: str) -> dict[str, list[HumanMessage]]:
    return {"messages": [HumanMessage(content=prompt)]}


async def test_plot_is_isolated_by_user_and_session() -> None:
    store = InMemoryStore()
    graph = create_persistent_plot_graph(store)
    context = PersistentPlotContext(user_id="user-1", session_id="thread-1")

    first = await graph.ainvoke(_state("Show the chart."), context=context)
    updated = await graph.ainvoke(_state("Set Q3 to 250."), context=context)
    reloaded = await graph.ainvoke(_state("Show the chart."), context=context)
    another_user = await graph.ainvoke(
        _state("Show the chart."),
        context=PersistentPlotContext(user_id="user-2", session_id="thread-1"),
    )
    another_thread = await graph.ainvoke(
        _state("Show the chart."),
        context=PersistentPlotContext(user_id="user-1", session_id="thread-2"),
    )

    assert "created revision 1" in first["messages"][-1].content
    assert "set Q3 to $250k in revision 2" in updated["messages"][-1].content
    assert "Q3 is highest at $250k" in reloaded["messages"][-1].content
    assert "loaded revision 2" in reloaded["messages"][-1].content
    assert "Q4 is highest at $230k" in another_user["messages"][-1].content
    assert "Q4 is highest at $230k" in another_thread["messages"][-1].content
    namespaces = await store.alist_namespaces(
        prefix=("demo", "persistent-plot", "threads")
    )
    assert len(namespaces) == 3
    items = [await store.aget(namespace, ARTIFACT_KEY) for namespace in namespaces]
    documents = [PlotDocument.model_validate(item.value) for item in items if item]
    assert sorted((document.q3, document.revision) for document in documents) == [
        (150, 1),
        (150, 1),
        (250, 2),
    ]


async def test_plot_can_be_reset() -> None:
    store = InMemoryStore()
    graph = create_persistent_plot_graph(store)
    context = PersistentPlotContext(user_id="user-1", session_id="thread-1")

    await graph.ainvoke(_state("Set Q2 to 300."), context=context)
    reset = await graph.ainvoke(_state("Reset the chart."), context=context)

    assert "Q4 is highest at $230k" in reset["messages"][-1].content
    assert "reset the chart to revision 1" in reset["messages"][-1].content
    namespaces = await store.alist_namespaces(
        prefix=("demo", "persistent-plot", "threads")
    )
    item = await store.aget(namespaces[0], ARTIFACT_KEY)
    assert item is not None
    assert PlotDocument.model_validate(item.value) == PlotDocument()


async def test_plot_stays_ephemeral_without_a_session_id() -> None:
    store = InMemoryStore()
    graph = create_persistent_plot_graph(store)

    result = await graph.ainvoke(
        _state("Show the chart."),
        context=PersistentPlotContext(user_id="user-1", session_id=None),
    )

    assert "both user and session_id are required" in result["messages"][-1].content
    assert await store.alist_namespaces() == []


async def test_plot_renders_a_non_streaming_completion(make_request) -> None:
    graph = create_persistent_plot_graph(InMemoryStore())
    registry = GraphRegistry(
        registry={
            "persistent-plot": create_persistent_plot_graph_config(lambda: graph),
        }
    )
    request = make_request(
        "persistent-plot",
        user="user-1",
        metadata={"session_id": "thread-1"},
    )

    invocation = await run_langgraph(
        request.model,
        request.messages,
        registry,
        request,
    )

    assert isinstance(invocation.output, str)
    assert "created revision 1" in invocation.output


async def test_plot_uses_the_openai_event_and_context_contract(make_request) -> None:
    graph = create_persistent_plot_graph(InMemoryStore())
    registry = GraphRegistry(
        registry={
            "persistent-plot": create_persistent_plot_graph_config(lambda: graph),
        }
    )
    request = make_request(
        "persistent-plot",
        user="user-1",
        metadata={
            "session_id": "thread-1",
            "langgraph_stream_events": "v1",
        },
    )

    stream = [
        item
        async for item in run_langgraph_stream(
            request.model,
            request.messages,
            registry,
            request,
        )
    ]
    events = [_public_event(item) for item in stream if isinstance(item, dict)]

    assert len(events) == 1
    assert events[0]["type"] == "artifact"
    assert events[0]["namespace"] == ["plots"]
    assert events[0]["data"]["kind"] == "plotly"
    assert events[0]["data"]["figure"]["data"][0]["y"] == [120, 180, 150, 230]
    assert "revision 1" in events[0]["data"]["figure"]["layout"]["title"]["text"]
    assert "created revision 1" in "".join(
        item for item in stream if isinstance(item, str)
    )
