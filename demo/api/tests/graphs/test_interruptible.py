import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from lgos_demo_api.graphs.interruptible import (
    create_interruptible_graph,
    output_to_message,
)


async def test_approved_refund_executes_and_notifies_customer(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    request = "Refund order ORDER-123"
    config = {"configurable": {"thread_id": "approved-refund"}}

    graph = create_interruptible_graph(sqlite_checkpointer)
    paused = await graph.ainvoke({"request": request}, config=config)
    interrupts = paused["__interrupt__"]

    assert len(interrupts) == 1
    review = interrupts[0]
    assert review.value == {
        "action": "refund",
        "question": "How should the refund be handled?",
        "request": request,
        "choices": ["approve", "reject"],
        "allow_other": True,
    }

    completed = await graph.ainvoke(
        Command(resume={review.id: "approve"}),
        config=config,
    )

    assert "__interrupt__" not in completed
    assert completed["review_outcome"] == "approve"
    assert completed["refund_executed"] is True
    assert completed["customer_notified"] is True
    assert output_to_message(completed).text == (
        f"Review workflow for: {request}\n"
        "- Refund: approve\n"
        "- Customer notification: sent\n"
        "- Executed actions: Refund, Customer notification"
    )


async def test_rejected_refund_skips_notification_and_execution(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    request = "Refund order ORDER-456"
    config = {"configurable": {"thread_id": "rejected-refund"}}
    graph = create_interruptible_graph(sqlite_checkpointer)
    first_pause = await graph.ainvoke({"request": request}, config=config)
    first_interrupt = first_pause["__interrupt__"][0]

    completed = await graph.ainvoke(
        Command(resume={first_interrupt.id: "reject"}),
        config=config,
    )

    assert "__interrupt__" not in completed
    assert completed["review_outcome"] == "reject"
    assert "refund_executed" not in completed
    assert "customer_notified" not in completed
    assert output_to_message(completed).text == (
        f"Review workflow for: {request}\n"
        "- Refund: reject\n"
        "- Customer notification: skipped\n"
        "- Executed actions: none"
    )


async def test_custom_refund_feedback_is_preserved_without_execution(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    config = {"configurable": {"thread_id": "reviewer-feedback"}}
    graph = create_interruptible_graph(sqlite_checkpointer)
    interrupted = await graph.ainvoke({"request": "Refund"}, config=config)
    feedback = "Please verify the delivery address first."
    responses = {item.id: feedback for item in interrupted["__interrupt__"]}

    completed = await graph.ainvoke(Command(resume=responses), config=config)

    assert completed["review_outcome"] == "feedback"
    assert completed["reviewer_feedback"] == feedback
    assert "refund_executed" not in completed
    assert output_to_message(completed).text == (
        "Review workflow for: Refund\n"
        "- Refund: feedback\n"
        "- Customer notification: skipped\n"
        "- Executed actions: none\n"
        f"- Reviewer feedback: {feedback}"
    )


async def test_rejects_an_empty_review_response(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    config = {"configurable": {"thread_id": "empty-response"}}
    graph = create_interruptible_graph(sqlite_checkpointer)
    interrupted = await graph.ainvoke({"request": "Refund"}, config=config)
    responses = {item.id: "   " for item in interrupted["__interrupt__"]}

    with pytest.raises(ValueError, match="non-empty string"):
        await graph.ainvoke(Command(resume=responses), config=config)
