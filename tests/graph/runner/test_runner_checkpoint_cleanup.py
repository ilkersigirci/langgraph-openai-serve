"""Cleanup of an interrupt run that ends without pausing."""

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from anyio import Event, fail_after, sleep, sleep_forever
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph

from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRegistry
from langgraph_openai_serve.graph import run as run_module
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream
from tests.graph.support.request import graph_request
from tests.graph.support.schemas import MessageState


class RecordingSaver(AsyncSqliteSaver):
    """Record thread deletions, which a test can make fail or outlast cleanup."""

    deleted_threads: list[str]
    delete_error: Exception | None = None
    slow_delete = False
    slow_delete_finished = False

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)
        if self.slow_delete:
            # Finite, not sleep_forever(): cleanup is shielded, so a lost
            # deadline would hang the test instead of failing it.
            await sleep(1)
            self.slow_delete_finished = True
        if self.delete_error is not None:
            raise self.delete_error
        await super().adelete_thread(thread_id)


@pytest.fixture
async def saver() -> AsyncIterator[RecordingSaver]:
    async with RecordingSaver.from_conn_string(":memory:") as saver:
        saver.deleted_threads = []
        yield saver


async def answer(_state: MessageState) -> dict[str, list[AIMessage]]:
    return {"messages": [AIMessage(content="done")]}


async def fail(_state: MessageState) -> dict[str, list[AIMessage]]:
    msg = "run failed"
    raise ValueError(msg)


async def fail_rendering(_output: Any) -> AIMessage:
    msg = "run failed"
    raise ValueError(msg)


def registry(
    saver: RecordingSaver,
    node: Callable[..., Any],
    *,
    run_coordinator: Any = None,
    **config: Any,
) -> GraphRegistry:
    graph = (
        StateGraph(MessageState)
        .add_node("run", node)
        .set_entry_point("run")
        .set_finish_point("run")
        .compile(checkpointer=saver)
    )
    return GraphRegistry(
        graphs={
            "cleanup": GraphConfig(
                graph=graph,
                description="DUMMY",
                features={GraphFeature.INTERRUPTS},
                **config,
            )
        },
        run_coordinator=run_coordinator or InMemoryRunCoordinator(),
    )


async def run(graph_registry: GraphRegistry) -> AIMessage:
    output = await run_langgraph(
        graph_request("cleanup"), [HumanMessage(content="hi")], graph_registry
    )
    assert isinstance(output, AIMessage)
    return output


@pytest.mark.parametrize(
    "delete_error",
    [None, RuntimeError("database unavailable")],
    ids=["cleanup-succeeds", "cleanup-fails"],
)
@pytest.mark.parametrize(
    ("node", "config"),
    [(answer, {"output_to_message": fail_rendering}), (fail, {})],
    ids=["rendering", "execution"],
)
async def test_failed_run_deletes_its_checkpoint_and_keeps_its_error(
    saver: RecordingSaver,
    node: Callable[..., Any],
    config: dict[str, Any],
    delete_error: Exception | None,
) -> None:
    saver.delete_error = delete_error

    with pytest.raises(ValueError, match="run failed"):
        await run(registry(saver, node, **config))

    assert len(saver.deleted_threads) == 1


@pytest.mark.parametrize("failure", ["preparation", "execution"])
async def test_failed_run_keeps_its_error_when_its_lease_release_fails(
    saver: RecordingSaver,
    failure: str,
) -> None:
    released = Event()

    @asynccontextmanager
    async def failing_release(_key: str) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            released.set()
            msg = "lease release failed"
            raise RuntimeError(msg)

    def fail_input(_request: Any, _messages: Any) -> None:
        msg = "run failed"
        raise ValueError(msg)

    graph_registry = (
        registry(
            saver, answer, run_coordinator=failing_release, request_to_input=fail_input
        )
        if failure == "preparation"
        else registry(saver, fail, run_coordinator=failing_release)
    )

    with pytest.raises(ValueError, match="run failed"):
        await run(graph_registry)

    assert released.is_set()


async def test_closing_stream_stops_the_graph_and_deletes_its_state(
    saver: RecordingSaver,
) -> None:
    model = FakeListChatModel(responses=["token"])
    stopped = Event()

    async def stream_then_wait(state: MessageState) -> None:
        try:
            await model.ainvoke(state["messages"])
            await sleep_forever()
        finally:
            stopped.set()

    events = run_langgraph_stream(
        graph_request("cleanup"),
        [HumanMessage(content="hi")],
        registry(saver, stream_then_wait),
    )
    with fail_after(1):
        assert await anext(events) == "t"
        await events.aclose()

    assert stopped.is_set()
    assert len(saver.deleted_threads) == 1


async def test_cleanup_failure_fails_a_successful_run(saver: RecordingSaver) -> None:
    saver.delete_error = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        await run(registry(saver, answer))


@pytest.mark.parametrize(
    ("node", "expected"),
    [(answer, TimeoutError), (fail, ValueError)],
    ids=["successful-run", "failed-run"],
)
async def test_hung_cleanup_is_abandoned_and_releases_the_lease(
    monkeypatch: pytest.MonkeyPatch,
    saver: RecordingSaver,
    node: Callable[..., Any],
    expected: type[Exception],
) -> None:
    # The deadline has no public setting; shorten it rather than wait 10 s.
    monkeypatch.setattr(run_module, "_CLEANUP_TIMEOUT", 0.01)
    saver.slow_delete = True
    released = Event()

    @asynccontextmanager
    async def lease(_key: str) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            released.set()

    with pytest.raises(expected):
        await run(registry(saver, node, run_coordinator=lease))

    assert not saver.slow_delete_finished
    assert released.is_set()
