"""Run LangGraph workflows from protocol-neutral requests and messages."""

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import (
    Command,
    CustomStreamPart,
    Durability,
    Interrupt,
    StreamMode,
    StreamPart,
    UpdatesStreamPart,
)

from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphRegistry,
)
from langgraph_openai_serve.graph.interrupt import (
    models as interrupt_models,
    state as interrupt_state,
)
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    build_runnable_config,
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


class BackgroundCheckpointIncompleteError(RuntimeError):
    """Raised when finalization finds no complete checkpointed output."""


@dataclass(frozen=True, slots=True)
class BackgroundGraphResult:
    """Checkpointed output, or pending interrupts, of one background operation."""

    output: LangGraphOutput
    root_messages: tuple[BaseMessage, ...]
    # Only callbacks of this delivery attempt report this usage.
    attempt_usage: UsageMetadata | None


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
    if run.pending_batch is not None:
        return run.pending_batch

    run.begin_execution()
    result = await run.graph.ainvoke(
        run.inputs,
        config=run.runnable_config,
        context=run.context,
        output_keys=run.graph.output_channels,
        durability=_durability(run),
        version="v2",
    )

    run.require_owner()
    interrupt_batch = _commit_interrupts(run, result.interrupts)
    if interrupt_batch is not None:
        return interrupt_batch

    return _with_usage(
        await run.config.render_output(result.value),
        run,
    )


async def run_background_graph(  # ruff: ignore[too-many-arguments] - A background run needs its checkpoint, interrupt identity, and answer.
    request: GraphRequest,
    messages: list[BaseMessage],
    config: GraphConfig,
    *,
    checkpoint_thread_id: str,
    run_id: str,
    resume: interrupt_models.InterruptResume | None,
    finalize_only: bool,
) -> BackgroundGraphResult:
    """Start, answer, recover, or only render one synchronously checkpointed run."""
    graph = await config.resolve_graph()
    usage_callback = UsageMetadataCallbackHandler()
    runnable_config = build_runnable_config(
        config.runtime_callbacks,
        configurable={"thread_id": checkpoint_thread_id},
        metadata={"lgos.model": request.model},
        extra_callbacks=[usage_callback],
    )
    if runnable_config is None:
        msg = "Background execution requires checkpoint runnable configuration."
        raise RuntimeError(msg)

    snapshot = await graph.aget_state(runnable_config, subgraphs=True)
    batch = _interrupt_batch(config, snapshot.interrupts, run_id)
    # An answer applies only while its interrupts are pending, so a retried
    # delivery continues from the checkpoint instead of answering twice.
    answer = (
        Command(resume=resume.values)
        if resume is not None
        and batch is not None
        and set(resume.values) == {item.id for item in batch.interrupts}
        else None
    )
    # A run paused at unanswered interrupts is published again, not advanced.
    if batch is not None and answer is None:
        return _background_result(batch, snapshot.values, usage_callback)
    # Pending task writes can make `next` empty before the super-step checkpoint
    # commits. LangGraph must resume those tasks to schedule downstream nodes.
    checkpoint_exists = snapshot.created_at is not None
    if checkpoint_exists and not snapshot.next and not snapshot.tasks:
        output = await config.render_output(
            await _background_output(graph, runnable_config)
        )
        return _background_result(output, snapshot.values, usage_callback)
    if finalize_only:
        msg = "No complete checkpointed graph output is available for finalization."
        raise BackgroundCheckpointIncompleteError(msg)

    if answer is not None:
        inputs = answer
    elif checkpoint_exists:
        inputs = None
    elif resume is None:
        inputs = await config.build_input(request, messages)
    else:
        msg = "The interrupted run no longer has a checkpoint."
        raise BackgroundCheckpointIncompleteError(msg)
    context = await config.build_context(request, graph)
    result = await graph.ainvoke(
        inputs,
        config=runnable_config,
        context=context,
        output_keys=graph.output_channels,
        durability="sync",
        version="v2",
    )
    completed = await graph.aget_state(runnable_config, subgraphs=True)
    if batch := _interrupt_batch(config, result.interrupts, run_id):
        return _background_result(batch, completed.values, usage_callback)
    # Static breakpoints return without interrupts, so confirm the checkpoint
    # head is complete before publishing its output.
    if completed.next or completed.tasks or completed.interrupts:
        msg = "Graph execution ended without complete checkpointed output."
        raise BackgroundCheckpointIncompleteError(msg)
    output = await config.render_output(result.value)
    return _background_result(output, completed.values, usage_callback)


def _background_result(
    output: LangGraphOutput,
    state: Any,
    usage_callback: UsageMetadataCallbackHandler,
) -> BackgroundGraphResult:
    attempt_usage = None
    for usage in usage_callback.usage_metadata.values():
        attempt_usage = add_usage(attempt_usage, usage)
    return BackgroundGraphResult(
        output=output,
        root_messages=root_messages(state),
        attempt_usage=attempt_usage,
    )


async def _background_output(
    graph: CompiledStateGraph,
    runnable_config: RunnableConfig,
) -> Any:
    """Read a completed checkpoint through LangGraph's public output path."""
    result = await graph.ainvoke(
        None,
        config=runnable_config,
        output_keys=graph.output_channels,
        durability="sync",
        version="v2",
    )
    return result.value


def root_messages(output: Any) -> tuple[BaseMessage, ...]:
    """Return the root ``messages`` channel of graph state or output."""
    values = (
        output.get("messages")
        if isinstance(output, Mapping)
        else getattr(output, "messages", None)
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(message for message in values if isinstance(message, BaseMessage))


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
    if run.pending_batch is not None:
        yield run.pending_batch
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
            run.require_owner()
            if part["type"] == "values":
                if not part["ns"]:
                    final_output = part["data"]
                    interrupts.extend(
                        item for item in part["interrupts"] if item not in interrupts
                    )
                continue
            visible_part = _visible_stream_part(part, stream_updates=stream_updates)
            if visible_part is not None:
                yield visible_part

    run.require_owner()
    interrupt_batch = _commit_interrupts(run, tuple(interrupts))
    if interrupt_batch is not None:
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
    """Persist checkpointed foreground runs when they pause or exit."""
    return (
        "exit"
        if run.config.supports(GraphFeature.INTERRUPTS)
        or run.config.supports(GraphFeature.BACKGROUND)
        else None
    )


def _with_usage(message: AIMessage, run: GraphRun) -> AIMessage:
    run.require_owner()
    usage = run.usage_metadata()
    return message.model_copy(update={"usage_metadata": usage}) if usage else message


async def _render_stream_output(output: Any, run: GraphRun) -> AIMessage:
    if output is _MISSING:
        msg = "LangGraph stream completed without a final value."
        raise RuntimeError(msg)
    return _with_usage(await run.config.render_output(output), run)


def _interrupt_batch(
    config: GraphConfig,
    interrupts: tuple[Interrupt, ...],
    run_id: str | None,
) -> interrupt_models.LangGraphInterruptBatch | None:
    if not interrupts:
        return None
    if not config.supports(GraphFeature.INTERRUPTS):
        msg = "Graphs using interrupt() must declare GraphFeature.INTERRUPTS."
        raise GraphConfigurationError(msg)
    return interrupt_state.interrupt_batch(interrupts, run_id)


def _commit_interrupts(
    run: GraphRun,
    interrupts: tuple[Interrupt, ...],
) -> interrupt_models.LangGraphInterruptBatch | None:
    batch = _interrupt_batch(run.config, interrupts, run.run_id)
    if batch is None:
        return None
    if (
        isinstance(run.inputs, Command)
        and isinstance(run.inputs.resume, dict)
        and set(run.inputs.resume).intersection(item.id for item in interrupts)
    ):
        msg = "A graph node may call interrupt() only once per invocation."
        raise GraphConfigurationError(msg)
    run.commit_interrupts()
    return batch
