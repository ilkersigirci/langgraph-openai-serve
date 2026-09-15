from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import pytest
from anyio import Event, fail_after, sleep_forever
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import GraphOutput, StreamPart, ValuesStreamPart

from langgraph_openai_serve import GraphConfig, GraphFeature
from langgraph_openai_serve.graph.runner import invoke_run, stream_run
from langgraph_openai_serve.graph.utils import GraphRun

THREAD_ID = "checkpoint-cleanup-thread"


class RecordingCheckpointer:
    def __init__(self, delete_error: Exception | None = None) -> None:
        self.deleted_threads: list[str] = []
        self._delete_error = delete_error

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)
        if self._delete_error is not None:
            raise self._delete_error


class CleanupGraph:
    output_channels = ("answer",)

    def __init__(
        self,
        events: Callable[[], AsyncIterator[StreamPart[Any, Any]]],
        *,
        delete_error: Exception | None = None,
    ) -> None:
        self._events = events
        self.checkpointer = RecordingCheckpointer(delete_error)

    def astream(self, *_args, **_kwargs) -> AsyncIterator[StreamPart[Any, Any]]:
        return self._events()

    async def ainvoke(self, *_args, **_kwargs) -> GraphOutput[Any]:
        output = None
        async for event in self._events():
            if event.get("type") == "values" and not event.get("ns"):
                output = event["data"]
        return GraphOutput(value=output)


def cleanup_run(
    graph: CleanupGraph,
    *,
    output_to_message: Callable[[Any], Any] | None = None,
    streamable_node_names: list[str] | None = None,
    resources: AsyncExitStack | None = None,
) -> GraphRun:
    return GraphRun(
        config=GraphConfig(
            graph=lambda: graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            output_to_message=output_to_message,
            streamable_node_names=streamable_node_names or [],
        ),
        graph=cast("Any", graph),
        inputs={},
        context=None,
        runnable_config={"configurable": {"thread_id": THREAD_ID}},
        run_id="11111111-1111-4111-8111-111111111111",
        checkpoint_thread_id=THREAD_ID,
        _resources=resources or AsyncExitStack(),
    )


@pytest.mark.parametrize(
    "delete_error",
    [None, RuntimeError("database unavailable")],
    ids=["cleanup-succeeds", "cleanup-fails"],
)
async def test_rendering_failure_deletes_without_replacing_error(
    delete_error: Exception | None,
) -> None:
    async def events():
        yield ValuesStreamPart(
            type="values",
            ns=(),
            data={"answer": "done"},
            interrupts=(),
        )

    async def fail_rendering(_output: Any) -> AIMessage:
        msg = "rendering failed"
        raise ValueError(msg)

    graph = CleanupGraph(events, delete_error=delete_error)

    run = cleanup_run(graph, output_to_message=fail_rendering)
    with pytest.raises(ValueError, match="rendering failed"):
        async with run:
            await invoke_run(run)

    assert graph.checkpointer.deleted_threads == [THREAD_ID]


@pytest.mark.parametrize(
    "delete_error",
    [None, RuntimeError("database unavailable")],
    ids=["cleanup-succeeds", "cleanup-fails"],
)
async def test_execution_failure_deletes_without_replacing_error(
    delete_error: Exception | None,
) -> None:
    async def events():
        yield ValuesStreamPart(
            type="values",
            ns=(),
            data={"answer": "partial"},
            interrupts=(),
        )
        msg = "graph failed"
        raise ValueError(msg)

    graph = CleanupGraph(events, delete_error=delete_error)

    run = cleanup_run(graph)
    with pytest.raises(ValueError, match="graph failed"):
        async with run:
            await invoke_run(run)

    assert graph.checkpointer.deleted_threads == [THREAD_ID]


async def test_closing_stream_deletes_incomplete_state_without_interrupts() -> None:
    closed = Event()

    async def events():
        try:
            yield {
                "type": "messages",
                "ns": (),
                "data": (
                    AIMessageChunk(content="token"),
                    {"langgraph_node": "generate"},
                ),
            }
            await sleep_forever()
        finally:
            closed.set()

    graph = CleanupGraph(events)
    run = cleanup_run(graph, streamable_node_names=["generate"])

    async with run:
        stream = stream_run(run)
        assert await anext(stream) == "token"
        with fail_after(1):
            await stream.aclose()

    assert closed.is_set()
    assert graph.checkpointer.deleted_threads == [THREAD_ID]


@pytest.mark.parametrize("stream", [False, True], ids=["invoke", "stream"])
async def test_successful_cleanup_failure_replaces_result(stream: bool) -> None:
    async def events():
        yield ValuesStreamPart(
            type="values",
            ns=(),
            data={"answer": "done"},
            interrupts=(),
        )

    graph = CleanupGraph(events, delete_error=RuntimeError("database unavailable"))
    run = cleanup_run(
        graph,
        output_to_message=lambda output: AIMessage(content=output["answer"]),
    )

    async def execute() -> None:
        async with run:
            if stream:
                _ = [event async for event in stream_run(run)]
            else:
                await invoke_run(run)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await execute()

    assert graph.checkpointer.deleted_threads == [THREAD_ID]


async def test_active_failure_wins_and_releases_resources_once() -> None:
    releases = 0

    async def events():
        msg = "graph failed"
        raise ValueError(msg)
        yield  # pragma: no cover

    @asynccontextmanager
    async def failing_lease():
        nonlocal releases
        try:
            yield
        finally:
            releases += 1
            msg = "lease release failed"
            raise RuntimeError(msg)

    resources = AsyncExitStack()
    await resources.enter_async_context(failing_lease())
    graph = CleanupGraph(events, delete_error=RuntimeError("database unavailable"))
    run = cleanup_run(graph, resources=resources)

    with pytest.raises(ValueError, match="graph failed"):
        async with run:
            await invoke_run(run)
    await run.aclose()

    assert graph.checkpointer.deleted_threads == [THREAD_ID]
    assert releases == 1
