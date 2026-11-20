"""Run LangGraph workflows from protocol-neutral requests and messages."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from typing import Any, Generic, TypeVar, cast

from anyio import CancelScope, create_memory_object_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.types import (
    CustomStreamPart,
    Interrupt,
    StreamMode,
    StreamPart,
    UpdatesStreamPart,
    ValuesStreamPart,
)
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan

from langgraph_openai_serve.graph import interrupt
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.graph.run import GraphRun, prepare_run

LangGraphOutput = AIMessage | interrupt.LangGraphInterruptBatch
"""A run's result: its final message, or the interrupts that paused it."""

LangGraphStreamEvent = str | LangGraphOutput | CustomStreamPart | UpdatesStreamPart
"""One streamed item: a text delta, a native LangGraph part, or the run's result."""

_Item = TypeVar("_Item")


class TaskStream(Generic[_Item]):
    """
    Produce an async generator's items from a dedicated asyncio task.

    AnyIO cancels a scope again at every await, and LangGraph then leaves node
    tasks running once it runs them in the background: in parallel steps, or
    while a stream reads message or custom events or subgraphs. The producer
    task instead receives exactly one asyncio cancellation, however its consumer
    is cancelled.
    """

    def __init__(self, source: AsyncGenerator[_Item, None], *, name: str) -> None:
        # An unbuffered handoff propagates consumer backpressure into the source.
        self._send, self.receive = create_memory_object_stream[_Item](max_buffer_size=0)

        async def produce() -> None:
            async with self._send, aclosing(source):
                async for item in source:
                    await self._send.send(item)

        self._producer = asyncio.create_task(produce(), name=name)

    async def aclose(self) -> None:
        """Stop the producer, then raise the exception that ended it, if any."""
        with CancelScope(shield=True):
            try:
                self._producer.cancel()
                # Like asyncio.TaskGroup, also ignore a producer that cancelled
                # itself: re-raised, that would read as the consumer's cancellation.
                # LangGraph raises NodeCancelledError for a node that cancels itself.
                with suppress(asyncio.CancelledError):
                    await self._producer
            finally:
                self._send.close()
                self.receive.close()


async def run_langgraph(
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_registry: GraphRegistry,
    *,
    resume: interrupt.InterruptResume | None = None,
    checkpoint_scope: str = "default",
) -> LangGraphOutput:
    """
    Prepare, execute, and close one graph run for direct Python callers.

    Args:
        request: Normalized graph selection, metadata, user, and client tools.
        messages: Decoded LangChain messages to process through the graph.
        graph_registry: The GraphRegistry instance containing registered graphs.
        resume: A decoded, complete interrupt answer batch, when resuming.
        checkpoint_scope: Server-trusted scope used to isolate checkpoint state.

    Returns:
        The final assistant message or the pending interrupt batch.

    """
    run = await prepare_run(
        request,
        messages,
        graph_registry,
        resume=resume,
        checkpoint_scope=checkpoint_scope,
    )
    return await collect_run(run)


async def run_langgraph_stream(
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_registry: GraphRegistry,
    *,
    resume: interrupt.InterruptResume | None = None,
    checkpoint_scope: str = "default",
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """
    Prepare, stream, and close one graph run for direct Python callers.

    Yields:
        Assistant text chunks, custom events, then the final output.

    """
    run = await prepare_run(
        request,
        messages,
        graph_registry,
        resume=resume,
        checkpoint_scope=checkpoint_scope,
    )
    async with aclosing(stream_run(run)) as events:
        async for event in events:
            yield event


async def collect_run(run: GraphRun) -> LangGraphOutput:
    """Execute a prepared run and return only its final output."""
    events = stream_run(run, streaming=False)
    async with aclosing(events):
        async for event in events:
            if isinstance(event, (AIMessage, interrupt.LangGraphInterruptBatch)):
                return event
    msg = "LangGraph run completed without a final output."
    raise RuntimeError(msg)


async def stream_run(
    run: GraphRun,
    *,
    streaming: bool = True,
    stream_updates: bool = False,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """
    Execute and close a prepared run, ending with its final output.

    Args:
        run: The prepared run.
        streaming: Yield assistant text and custom events, including those of
            nested subgraphs.
        stream_updates: Yield node updates; with ``streaming``, nested subgraph
            updates are yielded too.

    Yields:
        The requested intermediate events, then the final ``AIMessage`` or
        ``LangGraphInterruptBatch``.

    """
    # Without streaming, request only root values, exactly like ainvoke().
    stream_mode: list[StreamMode] = ["values"]
    if streaming:
        stream_mode += ["messages", "custom"]
    if stream_updates:
        stream_mode.append("updates")

    async with run.execution() as span:
        root_values: ValuesStreamPart[Any] | None = None
        interrupts: dict[str, Interrupt] = {}
        # LangGraph implements astream as an async generator, while its overload
        # returns AsyncIterator. Keep the concrete type so cancellation closes it.
        graph_stream = cast(
            "AsyncGenerator[StreamPart[Any, Any], None]",
            run.graph.astream(
                run.inputs,
                config=run.runnable_config,
                context=run.context,
                stream_mode=stream_mode,
                subgraphs=streaming,
                # Like ainvoke(), return the output schema's channels; astream
                # defaults to every state channel.
                output_keys=run.graph.output_channels,
                # Persist interrupt runs only when they pause or exit.
                durability="exit" if run.interrupt is not None else None,
                version="v2",
            ),
        )
        # Graph work runs under the span's context, not the span itself: its spans
        # are children, while code inside the graph, such as Langfuse's handler,
        # cannot write onto the span. asyncio copies this context into the
        # producer task, so the consumer's context does not change.
        graph_parent = NonRecordingSpan(span.get_span_context())
        with trace.use_span(graph_parent), run.langfuse_attributes():
            parts = TaskStream(graph_stream, name="langgraph-astream")
        async with aclosing(parts):
            async for part in parts.receive:
                if part["type"] == "values":
                    if not part["ns"]:
                        root_values = part
                        interrupts.update(
                            (item.id, item) for item in part["interrupts"]
                        )
                elif (event := _visible_event(part)) is not None:
                    yield event

        output: LangGraphOutput
        if interrupts:
            output = run.pause(tuple(interrupts.values()))
        elif root_values is None:
            msg = "LangGraph stream completed without a final value."
            raise RuntimeError(msg)
        else:
            with trace.use_span(graph_parent):
                message = await run.config.render_output(root_values["data"])
            usage = run.usage_metadata()
            output = (
                message.model_copy(update={"usage_metadata": usage})
                if usage
                else message
            )
    # The run is closed before its output is yielded: a consumer can resume a
    # paused run at once, and its handling is not part of the workflow.
    yield output


def _visible_event(part: StreamPart[Any, Any]) -> LangGraphStreamEvent | None:
    if part["type"] == "messages":
        message = part["data"][0]
        if isinstance(message, AIMessageChunk) and message.text:
            return str(message.text)
        return None
    if part["type"] in {"custom", "updates"}:
        return cast("CustomStreamPart | UpdatesStreamPart", part)
    return None


__all__ = [
    "LangGraphOutput",
    "LangGraphStreamEvent",
    "run_langgraph",
    "run_langgraph_stream",
]
