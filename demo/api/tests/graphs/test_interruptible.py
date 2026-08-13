import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from lgos_demo_api.graphs.interruptible import create_interruptible_graph


@pytest.mark.parametrize(
    ("decision", "outcome"),
    [
        pytest.param("approve", "Approved", id="approve"),
        pytest.param("reject", "Rejected", id="reject"),
    ],
)
async def test_resumes_with_the_selected_decision(
    sqlite_checkpointer: AsyncSqliteSaver,
    decision: str,
    outcome: str,
) -> None:
    request = "Refund order ORDER-123"
    config = {"configurable": {"thread_id": f"showcase-{decision}"}}

    graph = create_interruptible_graph(sqlite_checkpointer)
    interrupted = await graph.ainvoke({"request": request}, config=config)

    assert interrupted["__interrupt__"][0].value == {
        "question": "Approve this agent action?",
        "request": request,
        "choices": ["approve", "reject"],
    }

    resumed = await graph.ainvoke(Command(resume=decision), config=config)

    assert resumed["response"] == f"{outcome} agent action: {request}"


async def test_rejects_an_unknown_approval_decision(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    config = {"configurable": {"thread_id": "invalid-decision"}}
    graph = create_interruptible_graph(sqlite_checkpointer)
    await graph.ainvoke({"request": "Refund"}, config=config)

    with pytest.raises(ValueError, match=r"approve.*reject"):
        await graph.ainvoke(Command(resume="maybe"), config=config)
