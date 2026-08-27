import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from lgos_demo_api.graphs.interruptible import (
    APPROVAL_SPECS,
    create_interruptible_graph,
    output_to_message,
)


async def test_nested_approvals_pause_and_resume_as_one_parallel_batch(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    request = "Refund order ORDER-123"
    config = {"configurable": {"thread_id": "nested-approval-batch"}}

    graph = create_interruptible_graph(sqlite_checkpointer)
    interrupted = await graph.ainvoke({"request": request}, config=config)
    interrupts = interrupted["__interrupt__"]

    assert len(interrupts) == 2
    assert {item.value["question"] for item in interrupts} == {
        spec.question for spec in APPROVAL_SPECS
    }
    assert all(item.value["request"] == request for item in interrupts)
    assert all(item.value["choices"] == ["approve", "reject"] for item in interrupts)

    decisions = {
        item.id: (
            "approve" if item.value["question"] == "Approve the refund?" else "reject"
        )
        for item in interrupts
    }
    resumed = await graph.ainvoke(Command(resume=decisions), config=config)

    assert {
        (result["action"], result["decision"]) for result in resumed["approvals"]
    } == {
        ("Refund", "approve"),
        ("Customer notification", "reject"),
    }
    assert output_to_message(resumed).text == (
        f"Approval results for: {request}\n"
        "- Refund: approve\n"
        "- Customer notification: reject"
    )


async def test_rejects_an_unknown_approval_decision(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    config = {"configurable": {"thread_id": "invalid-decision"}}
    graph = create_interruptible_graph(sqlite_checkpointer)
    interrupted = await graph.ainvoke({"request": "Refund"}, config=config)
    decisions = {item.id: "maybe" for item in interrupted["__interrupt__"]}

    with pytest.raises(ValueError, match=r"approve.*reject"):
        await graph.ainvoke(Command(resume=decisions), config=config)
