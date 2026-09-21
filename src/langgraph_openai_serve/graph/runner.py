"""Run LangGraph workflows from protocol-neutral requests and messages."""

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import add_usage
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import (
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


class BackgroundGraphInterruptedError(RuntimeError):
    """Raised when a background graph reaches an unsupported interrupt."""


class BackgroundCheckpointIncompleteError(RuntimeError):
    """Raised when finalization finds no complete checkpointed output."""


@dataclass(frozen=True, slots=True)
class BackgroundGraphResult:
    """Checkpoint-reconstructible output from one background graph operation."""

    message: AIMessage
    root_messages: tuple[BaseMessage, ...]


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

    interrupt_batch = await _commit_interrupts(run, result.interrupts)
    if interrupt_batch is not None:
        return interrupt_batch

    return _with_usage(
        await run.config.render_output(result.value),
        run,
    )


async def run_background_graph(  # ruff: ignore[too-many-arguments] - Explicit checkpoint recovery boundary.
    request: GraphRequest,
    messages: list[BaseMessage],
    config: GraphConfig,
    *,
    checkpoint_thread_id: str,
    finalize_only: bool,
    initial_message_count: int,
) -> BackgroundGraphResult:
    """Start, recover, or only render one synchronously checkpointed operation."""
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
    checkpoint_exists = _snapshot_has_checkpoint(snapshot.config)
    if checkpoint_exists and not snapshot.next and not snapshot.interrupts:
        return await _background_result(
            config,
            graph,
            snapshot.values,
            usage_callback,
            initial_message_count=initial_message_count,
        )
    if snapshot.interrupts:
        msg = "Background execution does not support graph interrupts."
        raise BackgroundGraphInterruptedError(msg)
    if finalize_only:
        msg = "No complete checkpointed graph output is available for finalization."
        raise BackgroundCheckpointIncompleteError(msg)

    inputs = None
    if not checkpoint_exists:
        inputs = await config.build_input(request, messages)
    context = await config.build_context(request, graph)
    result = await graph.ainvoke(
        inputs,
        config=runnable_config,
        context=context,
        output_keys=graph.output_channels,
        durability="sync",
        version="v2",
    )
    if result.interrupts:
        msg = "Background execution does not support graph interrupts."
        raise BackgroundGraphInterruptedError(msg)

    # Rendering always uses the durable checkpoint head, including on the first
    # delivery, so publication recovery exercises the identical path.
    completed = await graph.aget_state(runnable_config, subgraphs=True)
    if (
        not _snapshot_has_checkpoint(completed.config)
        or completed.next
        or completed.interrupts
    ):
        msg = "Graph execution ended without complete checkpointed output."
        raise BackgroundCheckpointIncompleteError(msg)
    return await _background_result(
        config,
        graph,
        completed.values,
        usage_callback,
        initial_message_count=initial_message_count,
    )


def _snapshot_has_checkpoint(config: Mapping[str, Any] | None) -> bool:
    try:
        return get_checkpoint_id(cast("Any", config)) is not None
    except (AttributeError, KeyError, TypeError):
        return False


async def _background_result(
    config: GraphConfig,
    graph: CompiledStateGraph,
    state: Any,
    usage_callback: UsageMetadataCallbackHandler,
    *,
    initial_message_count: int,
) -> BackgroundGraphResult:
    message = await config.render_output(_checkpoint_output(graph, state))
    root_messages = _root_messages(state)
    total_usage = None
    for operation_message in root_messages[initial_message_count:]:
        if (
            isinstance(operation_message, AIMessage)
            and operation_message.usage_metadata is not None
        ):
            total_usage = add_usage(total_usage, operation_message.usage_metadata)
    if total_usage is None:
        for usage in usage_callback.usage_metadata.values():
            total_usage = add_usage(total_usage, usage)
    if total_usage is not None:
        message = message.model_copy(update={"usage_metadata": total_usage})
    return BackgroundGraphResult(
        message=message,
        root_messages=root_messages,
    )


def _checkpoint_output(graph: CompiledStateGraph, state: Any) -> Any:
    """Reconstruct the graph's declared v2 output from its full checkpoint state."""
    output_channels = graph.output_channels
    if isinstance(output_channels, str):
        output = (
            state[output_channels]
            if isinstance(state, Mapping) and output_channels in state
            else state
        )
    else:
        if not isinstance(state, Mapping):
            msg = "Checkpoint state does not expose the graph's output channels."
            raise BackgroundCheckpointIncompleteError(msg)
        output = {
            channel: state[channel] for channel in output_channels if channel in state
        }

    # LangGraph applies this mapper to v2 invocation output after selecting the
    # channels. Recovery must do the same for Pydantic and dataclass schemas.
    output_mapper = graph._output_mapper  # ruff: ignore[private-member-access] - Required for parity with LangGraph's v2 output coercion.
    return output_mapper(output) if output_mapper is not None else output


def _root_messages(output: Any) -> tuple[BaseMessage, ...]:
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

    interrupt_batch = await _commit_interrupts(run, tuple(interrupts))
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
        or run.config.background is not None
        else None
    )


def _with_usage(message: AIMessage, run: GraphRun) -> AIMessage:
    usage = run.usage_metadata()
    return message.model_copy(update={"usage_metadata": usage}) if usage else message


async def _render_stream_output(output: Any, run: GraphRun) -> AIMessage:
    if output is _MISSING:
        msg = "LangGraph stream completed without a final value."
        raise RuntimeError(msg)
    return _with_usage(await run.config.render_output(output), run)


async def _commit_interrupts(
    run: GraphRun,
    interrupts: tuple[Interrupt, ...],
) -> interrupt_models.LangGraphInterruptBatch | None:
    if not interrupts:
        return None
    if not run.config.supports(GraphFeature.INTERRUPTS):
        msg = "Graphs using interrupt() must declare GraphFeature.INTERRUPTS."
        raise GraphConfigurationError(msg)
    batch = await interrupt_state.durable_interrupt_batch(
        run.graph,
        interrupts,
        run.runnable_config,
        run.run_id,
    )
    if batch is not None:
        run.commit_interrupts()
    return batch
