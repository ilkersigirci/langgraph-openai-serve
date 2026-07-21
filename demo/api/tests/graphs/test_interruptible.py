from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from lgos_demo_api.graphs.interruptible import create_interruptible_graph


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("decision", "outcome"),
    [
        pytest.param("approve", "Approved", id="approve"),
        pytest.param("reject", "Rejected", id="reject"),
    ],
)
async def test_resumes_with_the_selected_decision(
    decision: str,
    outcome: str,
) -> None:
    request = "Refund order ORDER-123"
    config = {"configurable": {"thread_id": f"showcase-{decision}"}}

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = create_interruptible_graph(checkpointer)
        interrupted = await graph.ainvoke({"request": request}, config=config)

        assert interrupted["__interrupt__"][0].value == {
            "question": "Approve this agent action?",
            "request": request,
            "choices": ["approve", "reject"],
        }

        resumed = await graph.ainvoke(Command(resume=decision), config=config)

    assert resumed["response"] == f"{outcome} agent action: {request}"


@pytest.mark.anyio
async def test_checkpoint_survives_connection_reopen(tmp_path: Path) -> None:
    request = "Refund order ORDER-789"
    database = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "persisted-interrupt"}}

    async with AsyncSqliteSaver.from_conn_string(database) as checkpointer:
        graph = create_interruptible_graph(checkpointer)
        interrupted = await graph.ainvoke({"request": request}, config=config)
        assert interrupted["__interrupt__"]

    async with AsyncSqliteSaver.from_conn_string(database) as checkpointer:
        graph = create_interruptible_graph(checkpointer)
        resumed = await graph.ainvoke(Command(resume="approve"), config=config)

    assert resumed["response"] == f"Approved agent action: {request}"
