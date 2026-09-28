from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from http import HTTPStatus

import pytest
from langchain_core.messages import AIMessage, HumanMessage
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
from tests.graph.support.interrupt import (
    make_interrupt_graph,
    make_nested_multi_interrupt_graph,
)

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


async def test_a_run_answered_during_the_sweep_is_kept(
    make_request,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = GraphRegistry(
        graphs={
            MODEL: GraphConfig(
                # Both questions pause inside a subgraph.
                graph=make_nested_multi_interrupt_graph(sqlite_checkpointer),
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_message=lambda output: AIMessage(
                    content=",".join(output["answers"])
                ),
            )
        },
        run_coordinator=InMemoryRunCoordinator(),
    )
    first = await _pause(registry, make_request)
    answered: list[object] = []

    @asynccontextmanager
    async def answer_before_the_lease(_key: str) -> AsyncIterator[None]:
        # The user answers after the sweep listed the run as expired.
        resume = InterruptResume(
            run_id=first.run_id, values={first.interrupts[0].id: "yes"}
        )
        answered.append(
            await run_langgraph(make_request(MODEL), [], registry, resume=resume)
        )
        yield

    deleted = await delete_expired_interrupt_runs(
        sqlite_checkpointer, answer_before_the_lease, older_than=timedelta(0)
    )

    [second] = answered
    assert isinstance(second, LangGraphInterruptBatch)
    resume = InterruptResume(
        run_id=second.run_id, values={second.interrupts[0].id: "done"}
    )
    final = await run_langgraph(make_request(MODEL), [], registry, resume=resume)
    assert (deleted, final.content) == (0, "yes,done")
