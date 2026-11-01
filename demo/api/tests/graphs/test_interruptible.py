import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.interrupt import (
    InMemoryRunCoordinator,
    InterruptResume,
    LangGraphInterruptBatch,
)
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs.interruptible import (
    REVIEW_NOTICE,
    create_interruptible_graph,
    create_interruptible_graph_config,
    output_to_message,
)


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        pytest.param(
            " Approve ",
            "- Refund: approve\n"
            "- Customer notification: sent\n"
            "- Executed actions: Refund, Customer notification",
            id="approve",
        ),
        pytest.param(
            "reject",
            "- Refund: reject\n"
            "- Customer notification: skipped\n"
            "- Executed actions: none",
            id="reject",
        ),
        pytest.param(
            "Verify the address first.",
            "- Refund: feedback\n"
            "- Customer notification: skipped\n"
            "- Executed actions: none\n"
            "- Reviewer feedback: Verify the address first.",
            id="feedback",
        ),
    ],
)
async def test_review_answer_decides_which_actions_execute(
    sqlite_checkpointer: AsyncSqliteSaver,
    answer: str,
    outcome: str,
) -> None:
    request = "Refund order ORDER-123"
    config = {"configurable": {"thread_id": "refund-review"}}
    graph = create_interruptible_graph(sqlite_checkpointer)

    paused = await graph.ainvoke({"request": request}, config=config)
    (review,) = paused["__interrupt__"]
    assert review.value == {
        "action": "refund",
        "question": f"How should the refund be handled?\n\nRequest: {request}",
        "choices": ["approve", "reject"],
        "allow_other": True,
    }

    completed = await graph.ainvoke(Command(resume={review.id: answer}), config=config)

    assert "__interrupt__" not in completed
    assert output_to_message(completed).text == (
        f"Review workflow for: {request}\n{outcome}"
    )


async def test_rejects_an_empty_review_response(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    config = {"configurable": {"thread_id": "empty-response"}}
    graph = create_interruptible_graph(sqlite_checkpointer)
    interrupted = await graph.ainvoke({"request": "Refund"}, config=config)
    responses = {item.id: "   " for item in interrupted["__interrupt__"]}

    with pytest.raises(ValueError, match="must not be empty"):
        await graph.ainvoke(Command(resume=responses), config=config)


async def test_review_notice_streams_before_the_pause_only(
    make_graph_input,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    request = "Refund order ORDER-123"
    graph = create_interruptible_graph(sqlite_checkpointer)
    registry = GraphRegistry(
        graphs={"interruptible-approval": create_interruptible_graph_config(graph)},
        run_coordinator=InMemoryRunCoordinator(),
    )
    graph_request, messages = make_graph_input(
        "interruptible-approval", content=request
    )

    paused = [
        event async for event in run_langgraph_stream(graph_request, messages, registry)
    ]
    batch = paused[-1]
    assert isinstance(batch, LangGraphInterruptBatch)
    streamed = "".join(event for event in paused if isinstance(event, str))
    assert streamed == REVIEW_NOTICE.format(request=request)

    resume = InterruptResume(
        run_id=batch.run_id, values={batch.interrupts[0].id: "approve"}
    )
    resumed = [
        event
        async for event in run_langgraph_stream(
            graph_request, messages, registry, resume=resume
        )
    ]
    # Only the interrupted node reruns, so the notice is not repeated.
    assert not any(isinstance(event, str) for event in resumed)
    assert isinstance(resumed[-1], AIMessage)
    assert "- Refund: approve" in resumed[-1].text
