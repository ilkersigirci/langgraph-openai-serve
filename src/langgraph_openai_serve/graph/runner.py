"""Run LangGraph workflows from protocol-neutral requests and messages."""

from collections.abc import AsyncGenerator, Collection
from contextlib import aclosing
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.types import (
    Command,
    CustomStreamPart,
    Interrupt,
    StreamMode,
    StreamPart,
    UpdatesStreamPart,
)

from langgraph_openai_serve.core.errors import GraphError
from langgraph_openai_serve.graph import interrupt
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.graph.run import GraphRun, prepare_run
from langgraph_openai_serve.graph.telemetry import invoke_workflow

LangGraphOutput = AIMessage | interrupt.LangGraphInterruptBatch
LangGraphStreamEvent = str | LangGraphOutput | CustomStreamPart | UpdatesStreamPart

_MISSING = object()


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
    async with run:
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
    async with run:
        events = stream_run(run)
        async with aclosing(events):
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
    Execute a prepared run, ending with its final output.

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
    run.begin_execution()
    # Without streaming, request only root values, exactly like ainvoke().
    # LangGraph stops node tasks on one asyncio cancellation in every mode, but
    # AnyIO's repeated cancellation leaves them running once a stream reads
    # message or custom events or subgraphs; only token streams need those.
    stream_mode: list[StreamMode] = ["values"]
    if streaming:
        stream_mode += ["messages", "custom"]
    if stream_updates:
        stream_mode.append("updates")

    with invoke_workflow(run.request) as workflow:
        final_output: Any = _MISSING
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
                output_keys=run.graph.output_channels,
                # Persist interrupt runs only when they pause or exit.
                durability="exit" if run.interrupt is not None else None,
                version="v2",
            ),
        )
        async with aclosing(workflow.iterate(graph_stream)) as parts:
            async for part in parts:
                if part["type"] == "values":
                    if not part["ns"]:
                        final_output = part["data"]
                        interrupts.update(
                            (item.id, item) for item in part["interrupts"]
                        )
                elif (event := _visible_event(part)) is not None:
                    yield event

        output: LangGraphOutput
        if interrupts:
            output = _interrupt_batch(run, interrupts.values())
        elif final_output is _MISSING:
            msg = "LangGraph stream completed without a final value."
            raise RuntimeError(msg)
        else:
            with workflow.active():
                message = await run.config.render_output(final_output)
            usage = run.usage_metadata()
            output = (
                message.model_copy(update={"usage_metadata": usage})
                if usage
                else message
            )
    # The run ends with its output; the consumer's handling of it is not part of
    # the workflow.
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


def _interrupt_batch(
    run: GraphRun,
    interrupts: Collection[Interrupt],
) -> interrupt.LangGraphInterruptBatch:
    if run.interrupt is None:
        msg = "Graphs using interrupt() must declare GraphFeature.INTERRUPTS."
        raise GraphError(msg)
    # LangGraph gives every interrupt() of one node invocation the same ID, so
    # a second call would reuse the answered call ID and accept stale answers.
    if (
        isinstance(run.inputs, Command)
        and isinstance(run.inputs.resume, dict)
        and set(run.inputs.resume).intersection(item.id for item in interrupts)
    ):
        msg = "A graph node may call interrupt() only once per invocation."
        raise GraphError(msg)
    batch = interrupt.interrupt_batch(interrupts, run.interrupt.run_id)
    run.keep_checkpoint()
    return batch
