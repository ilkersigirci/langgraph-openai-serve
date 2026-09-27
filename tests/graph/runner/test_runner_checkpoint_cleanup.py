from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest
from anyio import Event, fail_after, sleep_forever
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import StreamPart, ValuesStreamPart

from langgraph_openai_serve import GraphConfig, GraphFeature
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.graph.run import GraphRun, InterruptRun
from langgraph_openai_serve.graph.runner import collect_run, stream_run

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


def cleanup_run(
    graph: CleanupGraph,
    *,
    output_to_message: Callable[[Any], Any] | None = None,
) -> GraphRun:
    return GraphRun(
        config=GraphConfig(
            graph=lambda: graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            output_to_message=output_to_message,
            run_coordinator=InMemoryRunCoordinator(),
        ),
        graph=cast("Any", graph),
        inputs={},
        context=None,
        runnable_config={"configurable": {"thread_id": THREAD_ID}},
        usage_callback=UsageMetadataCallbackHandler(),
        interrupt=InterruptRun(
            run_id="11111111-1111-4111-8111-111111111111",
            thread_id=THREAD_ID,
        ),
    )


def _values(answer: str) -> ValuesStreamPart:
    return ValuesStreamPart(
        type="values", ns=(), data={"answer": answer}, interrupts=()
    )


async def _fail_rendering(_output: Any) -> AIMessage:
    msg = "run failed"
    raise ValueError(msg)


async def _values_then_fail():
    yield _values("partial")
    msg = "run failed"
    raise ValueError(msg)


async def _values_only():
    yield _values("done")


@pytest.mark.parametrize(
    ("events", "output_to_message"),
    [(_values_only, _fail_rendering), (_values_then_fail, None)],
    ids=["rendering", "execution"],
)
async def test_failed_run_deletes_its_checkpoint(events, output_to_message) -> None:
    graph = CleanupGraph(events)
    run = cleanup_run(graph, output_to_message=output_to_message)

    with pytest.raises(ValueError, match="run failed"):
        async with run:
            await collect_run(run)

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
    run = cleanup_run(graph)

    async with run:
        stream = stream_run(run)
        assert await anext(stream) == "token"
        with fail_after(1):
            await stream.aclose()

    assert closed.is_set()
    assert graph.checkpointer.deleted_threads == [THREAD_ID]


async def test_cleanup_failure_fails_a_successful_run() -> None:
    graph = CleanupGraph(
        _values_only,
        delete_error=RuntimeError("database unavailable"),
    )
    run = cleanup_run(
        graph,
        output_to_message=lambda output: AIMessage(content=output["answer"]),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        async with run:
            await collect_run(run)

    assert graph.checkpointer.deleted_threads == [THREAD_ID]
