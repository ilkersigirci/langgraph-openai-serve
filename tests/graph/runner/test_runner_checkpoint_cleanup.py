from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anyio import Event, fail_after, sleep_forever
from langchain_core.messages import AIMessageChunk

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
        events: Callable[[], AsyncIterator[dict[str, Any]]],
        *,
        delete_error: Exception | None = None,
    ) -> None:
        self._events = events
        self.checkpointer = RecordingCheckpointer(delete_error)
        self.state_reads = 0

    def astream(self, *_args, **_kwargs) -> AsyncIterator[dict[str, Any]]:
        return self._events()

    async def aget_state(self, *_args, **_kwargs):
        self.state_reads += 1
        return SimpleNamespace(interrupts=())


def cleanup_run(
    graph: CleanupGraph,
    *,
    output_to_text: Callable[[Any], Any] | None = None,
    streamable_node_names: list[str] | None = None,
) -> GraphRun:
    return GraphRun(
        config=GraphConfig(
            graph=lambda: graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            output_to_text=output_to_text,
            streamable_node_names=streamable_node_names or [],
        ),
        graph=cast("Any", graph),
        inputs={},
        context=None,
        runnable_config={"configurable": {"thread_id": THREAD_ID}},
        run_id="11111111-1111-4111-8111-111111111111",
        checkpoint_thread_id=THREAD_ID,
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
        yield {"type": "values", "ns": (), "data": {"answer": "done"}}

    async def fail_rendering(_output: Any) -> str:
        raise ValueError("rendering failed")

    graph = CleanupGraph(events, delete_error=delete_error)

    with pytest.raises(ValueError, match="rendering failed"):
        await invoke_run(cleanup_run(graph, output_to_text=fail_rendering))

    assert graph.checkpointer.deleted_threads == [THREAD_ID]
    assert graph.state_reads == 1


@pytest.mark.parametrize(
    "delete_error",
    [None, RuntimeError("database unavailable")],
    ids=["cleanup-succeeds", "cleanup-fails"],
)
async def test_execution_failure_deletes_without_replacing_error(
    delete_error: Exception | None,
) -> None:
    async def events():
        yield {"type": "values", "ns": (), "data": {"answer": "partial"}}
        raise ValueError("graph failed")

    graph = CleanupGraph(events, delete_error=delete_error)

    with pytest.raises(ValueError, match="graph failed"):
        await invoke_run(cleanup_run(graph))

    assert graph.checkpointer.deleted_threads == [THREAD_ID]
    assert graph.state_reads == 0


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
    stream = stream_run(cleanup_run(graph, streamable_node_names=["generate"]))

    assert await anext(stream) == "token"
    with fail_after(1):
        await stream.aclose()

    assert closed.is_set()
    assert graph.checkpointer.deleted_threads == [THREAD_ID]
