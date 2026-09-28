from datetime import timedelta
from http import HTTPStatus

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    InvalidRequestError,
)
from langgraph_openai_serve.graph.interrupt import (
    InMemoryRunCoordinator,
    InterruptResume,
    LangGraphInterruptBatch,
    checkpoint_key,
    delete_expired_interrupt_runs,
)
from langgraph_openai_serve.graph.runner import run_langgraph
from tests.graph.support.interrupt import make_interrupt_graph

MODEL = "review"


@pytest.fixture
def coordinator() -> InMemoryRunCoordinator:
    return InMemoryRunCoordinator()


@pytest.fixture
def registry(
    sqlite_checkpointer: AsyncSqliteSaver, coordinator: InMemoryRunCoordinator
) -> GraphRegistry:
    return GraphRegistry(
        graphs={
            MODEL: GraphConfig(
                graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
            )
        },
        run_coordinator=coordinator,
    )


async def _pause(registry: GraphRegistry, make_request) -> LangGraphInterruptBatch:
    batch = await run_langgraph(
        make_request(MODEL), [HumanMessage(content="question")], registry
    )
    assert isinstance(batch, LangGraphInterruptBatch)
    return batch


async def test_expired_runs_are_deleted_and_can_no_longer_resume(
    registry: GraphRegistry,
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
    coordinator: InMemoryRunCoordinator,
) -> None:
    batch = await _pause(registry, make_request)

    fresh = await delete_expired_interrupt_runs(
        sqlite_checkpointer, coordinator, older_than=timedelta(hours=1)
    )
    expired = await delete_expired_interrupt_runs(
        sqlite_checkpointer, coordinator, older_than=timedelta(0)
    )

    assert (fresh, expired) == (0, 1)
    resume = InterruptResume(
        run_id=batch.run_id, values={batch.interrupts[0].id: "yes"}
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        await run_langgraph(make_request(MODEL), [], registry, resume=resume)
    assert (exc_info.value.status_code, exc_info.value.code) == (
        HTTPStatus.CONFLICT,
        "interrupt_state_conflict",
    )


async def test_runs_in_use_and_other_threads_are_kept(
    registry: GraphRegistry,
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
    coordinator: InMemoryRunCoordinator,
) -> None:
    batch = await _pause(registry, make_request)
    # A thread the application checkpoints itself, outside LGOS.
    application_thread = {"configurable": {"thread_id": "application-thread"}}
    await make_interrupt_graph(checkpointer=sqlite_checkpointer).ainvoke(
        {"messages": [HumanMessage(content="question")]}, application_thread
    )

    async with coordinator(checkpoint_key(MODEL, batch.run_id)):
        in_use = await delete_expired_interrupt_runs(
            sqlite_checkpointer, coordinator, older_than=timedelta(0)
        )
    released = await delete_expired_interrupt_runs(
        sqlite_checkpointer, coordinator, older_than=timedelta(0)
    )

    assert (in_use, released) == (0, 1)
    assert await sqlite_checkpointer.aget_tuple(application_thread) is not None
