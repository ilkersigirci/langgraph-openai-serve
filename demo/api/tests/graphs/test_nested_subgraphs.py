from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.interrupt import (
    InMemoryRunCoordinator,
    InterruptResume,
    LangGraphInterruptBatch,
)
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

from lgos_demo_api.graphs.nested_subgraphs import (
    create_nested_subgraphs_graph,
    create_nested_subgraphs_graph_config,
)


def _registry(checkpointer: AsyncSqliteSaver) -> GraphRegistry:
    graph = create_nested_subgraphs_graph(checkpointer)
    return GraphRegistry(
        graphs={"nested-subgraphs": create_nested_subgraphs_graph_config(graph)},
        run_coordinator=InMemoryRunCoordinator(),
    )


async def test_stream_crosses_both_levels_and_resumes_only_the_grandchild(
    make_graph_input,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = _registry(sqlite_checkpointer)
    graph_request, messages = make_graph_input(
        "nested-subgraphs", content="Quarterly risks"
    )

    paused = [
        event async for event in run_langgraph_stream(graph_request, messages, registry)
    ]

    batch = paused[-1]
    assert isinstance(batch, LangGraphInterruptBatch)
    assert [item.value["question"] for item in batch.interrupts] == [
        "Approve the child subgraph's draft?"
    ]
    assert [
        event["data"]["description"] for event in paused if isinstance(event, dict)
    ] == ["Drafting in the child subgraph"]
    assert "".join(event for event in paused if isinstance(event, str)) == (
        "Draft from the child subgraph: Quarterly risks"
    )

    resume = InterruptResume(
        run_id=batch.run_id, values={batch.interrupts[0].id: "approve"}
    )
    resumed = [
        event
        async for event in run_langgraph_stream(
            graph_request, messages, registry, resume=resume
        )
    ]

    # The child's draft node completed before the pause, so it does not rerun.
    assert not any(isinstance(event, dict) for event in resumed)
    streamed = "".join(event for event in resumed if isinstance(event, str))
    assert streamed == "The grandchild subgraph resumed with: approve"
    assert isinstance(resumed[-1], AIMessage)
    assert resumed[-1].text == streamed


async def test_non_streaming_run_pauses_and_resumes_like_background_work(
    make_graph_input,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = _registry(sqlite_checkpointer)
    graph_request, messages = make_graph_input(
        "nested-subgraphs", content="Quarterly risks"
    )

    paused = await run_langgraph(graph_request, messages, registry)

    assert isinstance(paused, LangGraphInterruptBatch)
    resume = InterruptResume(
        run_id=paused.run_id, values={paused.interrupts[0].id: "reject"}
    )
    completed = await run_langgraph(graph_request, messages, registry, resume=resume)
    assert isinstance(completed, AIMessage)
    assert completed.text == "The grandchild subgraph resumed with: reject"
