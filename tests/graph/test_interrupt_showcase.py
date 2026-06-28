import asyncio
import multiprocessing
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from demo.api.graphs.interruptible import create_interruptible_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command


async def _execute_checkpoint_phase(database: str, action: str) -> None:
    config = {"configurable": {"thread_id": "process-restart"}}

    async with AsyncSqliteSaver.from_conn_string(database) as checkpointer:
        graph = create_interruptible_graph(checkpointer)
        if action == "interrupt":
            result = await graph.ainvoke(
                {"request": "Refund order ORDER-999"},
                config=config,
            )
            assert result["__interrupt__"]
        elif action == "resume":
            result = await graph.ainvoke(Command(resume="approve"), config=config)
            assert result["response"] == "Approved agent action: Refund order ORDER-999"
        else:
            raise ValueError(f"Unknown checkpoint phase: {action}")


def _checkpoint_process(database: str, action: str) -> None:
    asyncio.run(_execute_checkpoint_phase(database, action))


def _run_checkpoint_process(database: Path, action: str) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_checkpoint_process,
        args=(str(database), action),
    )
    process.start()
    process.join(timeout=30)

    if process.is_alive():
        process.kill()
        process.join()
        process.close()
        pytest.fail(f"Checkpoint {action} process timed out")

    exitcode = process.exitcode
    process.close()
    assert exitcode == 0, f"Checkpoint {action} process exited with {exitcode}"


@pytest.fixture
async def interruptible_graph() -> AsyncIterator[CompiledStateGraph]:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield create_interruptible_graph(checkpointer)


@pytest.mark.anyio
async def test_interrupt_showcase_approval_resume(
    interruptible_graph: CompiledStateGraph,
) -> None:
    config = {"configurable": {"thread_id": "showcase-approve"}}

    first = await interruptible_graph.ainvoke(
        {"request": "Refund order ORDER-123"},
        config=config,
    )
    interrupt = first["__interrupt__"][0]

    assert interrupt.value == {
        "question": "Approve this agent action?",
        "request": "Refund order ORDER-123",
        "choices": ["approve", "reject"],
    }

    resumed = await interruptible_graph.ainvoke(
        Command(resume="approve"), config=config
    )

    assert resumed["response"] == "Approved agent action: Refund order ORDER-123"


@pytest.mark.anyio
async def test_interrupt_showcase_reject_resume(
    interruptible_graph: CompiledStateGraph,
) -> None:
    config = {"configurable": {"thread_id": "showcase-reject"}}

    await interruptible_graph.ainvoke(
        {"request": "Refund order ORDER-456"},
        config=config,
    )
    resumed = await interruptible_graph.ainvoke(Command(resume="reject"), config=config)

    assert resumed["response"] == "Rejected agent action: Refund order ORDER-456"


@pytest.mark.anyio
async def test_interrupt_checkpoint_persists_across_connections(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "persisted-interrupt"}}

    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        first_graph = create_interruptible_graph(checkpointer)
        await first_graph.ainvoke(
            {"request": "Refund order ORDER-789"},
            config=config,
        )

    async with AsyncSqliteSaver.from_conn_string(str(database)) as checkpointer:
        assert await checkpointer.aget_tuple(config) is not None

        resumed_graph = create_interruptible_graph(checkpointer)
        resumed = await resumed_graph.ainvoke(
            Command(resume="approve"),
            config=config,
        )

    assert resumed["response"] == "Approved agent action: Refund order ORDER-789"


def test_interrupt_checkpoint_persists_across_process_restarts(tmp_path: Path) -> None:
    database = tmp_path / "checkpoints.sqlite"

    for action in ("interrupt", "resume"):
        _run_checkpoint_process(database, action)
