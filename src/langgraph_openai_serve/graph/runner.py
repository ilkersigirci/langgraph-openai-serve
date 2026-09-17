"""Run LangGraph workflows from protocol-neutral requests and messages."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.types import (
    CustomStreamPart,
    Durability,
    Interrupt,
    StreamMode,
    StreamPart,
    UpdatesStreamPart,
)

from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import (
    models as interrupt_models,
    state as interrupt_state,
)
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    prepare_run,
)

LangGraphOutput = AIMessage | interrupt_models.LangGraphInterruptBatch
LangGraphStreamEvent = (
    str
    | AIMessage
    | interrupt_models.LangGraphInterruptBatch
    | CustomStreamPart
    | UpdatesStreamPart
)

_MISSING = object()


async def run_langgraph(
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_registry: GraphRegistry,
    *,
    resume: interrupt_models.InterruptResume | None = None,
    checkpoint_scope: str = "default",
) -> LangGraphOutput:
    """
    Prepare and invoke a graph for direct runner callers.

    This convenience wrapper combines :func:`prepare_run` and :func:`invoke_run`.
    The HTTP route prepares its run before creating a response so preparation
    errors can be returned as OpenAI-compatible HTTP errors; its service therefore
    calls ``invoke_run`` directly with that prepared run.

    Examples:
        >>> output = await run_langgraph(request, messages, registry)
        >>> print(output)

    Args:
        request: Normalized graph selection, metadata, user, and client tools.
        messages: Decoded LangChain messages to process through the graph.
        graph_registry: The GraphRegistry instance containing registered graphs.
        resume: A decoded, complete interrupt answer batch, when resuming.
        checkpoint_scope: Server-trusted scope used to isolate checkpoint state.

    Returns:
        The durable graph output.

    """
    run = await prepare_run(
        request,
        messages,
        graph_registry,
        resume=resume,
        checkpoint_scope=checkpoint_scope,
    )

    async with run:
        return await invoke_run(run)


async def invoke_run(run: GraphRun) -> LangGraphOutput:
    """Invoke a graph already owned by an active ``GraphRun`` context."""
    run.require_owner()
    if not run.should_execute:
        interrupt_batch = await _durable_interrupt_batch(
            run,
            run.pending_interrupts,
        )
        if interrupt_batch is None:
            msg = "Pending interrupt state disappeared before use."
            raise RuntimeError(msg)
        run.commit_interrupts()
        return interrupt_batch

    run.begin_execution()
    result = await run.graph.ainvoke(
        run.inputs,
        config=run.runnable_config,
        context=run.context,
        output_keys=run.graph.output_channels,
        durability=_durability(run),
        version="v2",
    )

    if run.config.supports(GraphFeature.INTERRUPTS):
        interrupt_batch = await _durable_interrupt_batch(run, result.interrupts)
        if interrupt_batch is not None:
            run.commit_interrupts()
            return interrupt_batch

    return _with_usage(
        await run.config.render_output(result.value),
        run,
    )


async def run_langgraph_stream(
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_registry: GraphRegistry,
    *,
    resume: interrupt_models.InterruptResume | None = None,
    checkpoint_scope: str = "default",
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """
    Prepare and stream a graph for direct runner callers.

    This convenience wrapper combines :func:`prepare_run` and :func:`stream_run`.
    The HTTP route prepares its run before starting the streaming response so
    preparation errors remain normal OpenAI-compatible HTTP errors; its service
    therefore calls ``stream_run`` directly with that prepared run.

    Args:
        request: Normalized graph selection, metadata, user, and client tools.
        messages: Decoded LangChain messages to process through the graph.
        graph_registry: The registry containing the graph configurations.
        resume: A decoded, complete interrupt answer batch, when resuming.
        checkpoint_scope: Server-trusted scope used to isolate checkpoint state.

    Yields:
        Assistant text chunks, custom events, or LangGraph interrupts.

    """
    run = await prepare_run(
        request,
        messages,
        graph_registry,
        resume=resume,
        checkpoint_scope=checkpoint_scope,
    )
    async with run:
        run_stream = stream_run(run)
        async with aclosing(run_stream):
            async for event in run_stream:
                yield event


async def stream_run(
    run: GraphRun,
    *,
    stream_messages: bool = True,
    stream_updates: bool = False,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """
    Stream a graph already owned by an active ``GraphRun`` context.

    Yields:
        LangGraph stream events.

    """
    run.require_owner()
    if not run.should_execute:
        interrupt_batch = await _durable_interrupt_batch(
            run,
            run.pending_interrupts,
        )
        if interrupt_batch is None:
            msg = "Pending interrupt state disappeared before use."
            raise RuntimeError(msg)
        run.commit_interrupts()
        yield interrupt_batch
        return

    run.begin_execution()
    final_output: Any = _MISSING
    interrupts: list[Interrupt] = []

    # LangGraph implements this as an async generator, while its overload
    # returns AsyncIterator. Keep the concrete type so cancellation closes it.
    graph_stream = cast(
        "AsyncGenerator[StreamPart[Any, Any], None]",
        run.graph.astream(
            run.inputs,
            config=run.runnable_config,
            context=run.context,
            stream_mode=_stream_modes(
                stream_messages=stream_messages,
                stream_updates=stream_updates,
            ),
            subgraphs=True,
            output_keys=run.graph.output_channels,
            durability=_durability(run),
            version="v2",
        ),
    )
    async with aclosing(graph_stream):
        async for part in graph_stream:
            if part["type"] == "values":
                if not part["ns"]:
                    final_output = part["data"]
                    interrupts.extend(part["interrupts"])
                continue
            visible_part = _visible_stream_part(part, stream_updates=stream_updates)
            if visible_part is not None:
                yield visible_part

    if run.config.supports(GraphFeature.INTERRUPTS):
        interrupt_batch = await _durable_interrupt_batch(run, tuple(interrupts))
        if interrupt_batch is not None:
            run.commit_interrupts()
            yield interrupt_batch
            return

    yield await _render_stream_output(final_output, run)


def _visible_stream_part(
    part: StreamPart[Any, Any],
    *,
    stream_updates: bool,
) -> LangGraphStreamEvent | None:
    if part["type"] == "custom":
        return part
    if part["type"] == "updates" and stream_updates:
        return part
    if part["type"] == "messages":
        message = part["data"][0]
        if isinstance(message, AIMessageChunk):
            return str(message.text) or None
    return None


def _stream_modes(
    *,
    stream_messages: bool,
    stream_updates: bool,
) -> list[StreamMode]:
    """Build the requested LangGraph stream modes."""
    stream_mode: list[StreamMode] = ["custom", "values"]
    if stream_messages:
        stream_mode.insert(0, "messages")
    if stream_updates:
        stream_mode.append("updates")
    return stream_mode


def _durability(run: GraphRun) -> Durability | None:
    """Persist interrupt runs when they pause or exit."""
    return "exit" if run.config.supports(GraphFeature.INTERRUPTS) else None


def _with_usage(message: AIMessage, run: GraphRun) -> AIMessage:
    usage = run.usage_metadata()
    return message.model_copy(update={"usage_metadata": usage}) if usage else message


async def _render_stream_output(output: Any, run: GraphRun) -> AIMessage:
    if output is _MISSING:
        msg = "LangGraph stream completed without a final value."
        raise RuntimeError(msg)
    return _with_usage(await run.config.render_output(output), run)


async def _durable_interrupt_batch(
    run: GraphRun,
    interrupts: tuple[Interrupt, ...],
) -> interrupt_models.LangGraphInterruptBatch | None:
    return await interrupt_state.durable_interrupt_batch(
        run.graph,
        interrupts,
        run.runnable_config,
        run.run_id,
    )
