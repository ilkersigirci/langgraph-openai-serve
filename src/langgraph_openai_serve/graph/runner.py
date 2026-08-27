"""Run LangGraph workflows behind the OpenAI-compatible chat API."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from anyio import CancelScope
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.constants import TAG_NOSTREAM
from langgraph.types import CustomStreamPart, GraphOutput, StreamMode

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import (
    models as interrupt_models,
    state as interrupt_state,
)
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    prepare_run,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = get_logger(__name__)


@dataclass(frozen=True)
class LangGraphInvocation:
    """A durable graph result."""

    output: "LangGraphOutput"


LangGraphOutput = AIMessage | interrupt_models.LangGraphInterruptBatch
LangGraphStreamEvent = (
    str | AIMessage | interrupt_models.LangGraphInterruptBatch | CustomStreamPart
)

_MISSING = object()
_CheckpointDisposition = Literal["unknown", "preserve", "delete"]


async def run_langgraph(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> LangGraphInvocation:
    """
    Prepare and invoke a graph for direct runner callers.

    This convenience wrapper combines :func:`prepare_run` and :func:`invoke_run`.
    The HTTP route prepares its run before creating a response so preparation
    errors can be returned as OpenAI-compatible HTTP errors; its service therefore
    calls ``invoke_run`` directly with that prepared run.

    Examples:
        >>> invocation = await run_langgraph("my-model", messages, registry)
        >>> print(invocation.output)

    Args:
        model: The name of the model to use, which also determines which graph to use.
        messages: A list of messages to process through the LangGraph.
        graph_registry: The GraphRegistry instance containing registered graphs.
        request: The complete chat completion request passed to graph adapters.

    Returns:
        The durable graph output.

    """
    run = await prepare_run(model, messages, graph_registry, request)

    return await invoke_run(run)


async def invoke_run(run: GraphRun) -> LangGraphInvocation:
    """Invoke a graph and return only its durable result."""
    checkpoint_disposition: _CheckpointDisposition = "unknown"
    try:
        if not run.should_execute:
            interrupt_batch = await _durable_interrupt_batch(run)
            if interrupt_batch is None:
                msg = "Pending interrupt state disappeared before use."
                raise RuntimeError(msg)
            checkpoint_disposition = "preserve"
            return LangGraphInvocation(output=interrupt_batch)

        result = cast(
            "GraphOutput[Any]",
            await run.graph.ainvoke(
                run.inputs,
                config=run.runnable_config,
                context=run.context,
                output_keys=run.graph.output_channels,
                **_invoke_options(run),
            ),
        )

        if run.config.supports(GraphFeature.INTERRUPTS):
            interrupt_batch = await _durable_interrupt_batch(run)
            if interrupt_batch is not None:
                checkpoint_disposition = "preserve"
                return LangGraphInvocation(output=interrupt_batch)

        rendered_output = _with_usage(
            await run.config.render_output(result.value),
            run,
        )
        if run.config.supports(GraphFeature.INTERRUPTS):
            checkpoint_disposition = "delete"

        return LangGraphInvocation(output=rendered_output)
    finally:
        await finalize_run(run, checkpoint_disposition)


async def run_langgraph_stream(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """
    Prepare and stream a graph for direct runner callers.

    This convenience wrapper combines :func:`prepare_run` and :func:`stream_run`.
    The HTTP route prepares its run before starting the streaming response so
    preparation errors remain normal OpenAI-compatible HTTP errors; its service
    therefore calls ``stream_run`` directly with that prepared run.

    Args:
        model: The name of the model (graph) to run.
        messages: A list of OpenAI-compatible messages.
        graph_registry: The registry containing the graph configurations.
        request: The complete chat completion request passed to graph adapters.

    Yields:
        Assistant text chunks, custom events, or LangGraph interrupts.

    """
    run = await prepare_run(model, messages, graph_registry, request)
    run_stream = stream_run(run)
    async with aclosing(run_stream):
        async for event in run_stream:
            yield event


async def stream_run(
    run: GraphRun,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """
    Stream an already prepared LangGraph invocation.

    Yields:
        LangGraph stream events.

    """
    checkpoint_disposition: _CheckpointDisposition = "unknown"
    try:
        if not run.should_execute:
            interrupt_batch = await _durable_interrupt_batch(run)
            if interrupt_batch is None:
                msg = "Pending interrupt state disappeared before use."
                raise RuntimeError(msg)
            checkpoint_disposition = "preserve"
            yield interrupt_batch
            return

        stream_mode: list[StreamMode] = ["messages", "custom", "values"]
        final_output: Any = _MISSING

        graph_stream = cast(
            "AsyncGenerator[dict[str, Any], None]",
            run.graph.astream(
                run.inputs,
                config=run.runnable_config,
                context=run.context,
                stream_mode=stream_mode,
                **_astream_options(run),
            ),
        )
        async with aclosing(graph_stream):
            async for event in graph_stream:
                if event.get("type") == "custom":
                    yield cast("CustomStreamPart", event)
                    continue

                if event.get("type") == "values" and not event.get("ns"):
                    final_output = event.get("data")
                    continue

                if event.get("type") != "messages":
                    continue

                content = text_from_message_event(event, run)
                if content:
                    yield content

        if run.config.supports(GraphFeature.INTERRUPTS):
            interrupt_batch = await _durable_interrupt_batch(run)
            if interrupt_batch is not None:
                checkpoint_disposition = "preserve"
                yield interrupt_batch
                return
            else:
                checkpoint_disposition = "delete"

        yield await _render_stream_output(final_output, run)
    finally:
        await finalize_run(run, checkpoint_disposition)


def text_from_message_event(event: dict, run: GraphRun) -> str | None:
    """Extract visible text from a streamable LangGraph message event."""
    message, metadata = event["data"]
    if not isinstance(message, AIMessageChunk):
        return None
    if TAG_NOSTREAM in (metadata.get("tags") or []):
        return None
    if metadata.get("langgraph_node") not in run.config.streamable_node_names:
        return None

    content = str(message.text)
    return content or None


def _invoke_options(run: GraphRun) -> dict[str, Any]:
    """Build shared LangGraph invocation options."""
    options: dict[str, Any] = {"version": "v2"}
    if run.config.supports(GraphFeature.INTERRUPTS):
        options["durability"] = "exit"
    return options


def _astream_options(run: GraphRun) -> dict[str, Any]:
    """Build LangGraph streaming options."""
    return {"subgraphs": True, **_invoke_options(run)}


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
) -> interrupt_models.LangGraphInterruptBatch | None:
    return await interrupt_state.durable_interrupt_batch(
        run.graph,
        run.runnable_config,
        run.run_id,
    )


async def finalize_run(
    run: GraphRun,
    checkpoint_disposition: _CheckpointDisposition,
) -> None:
    """
    Finalize checkpoint retention, then release the run lease.

    Only state exposed as a resumable interrupt is preserved. Cleanup for an
    unclassified run is best-effort so it cannot mask the failure that prevented
    classification.
    """
    with CancelScope(shield=True):
        try:
            if checkpoint_disposition == "delete" or (
                checkpoint_disposition == "unknown"
                and run.config.supports(GraphFeature.INTERRUPTS)
            ):
                await delete_checkpoint_thread(run)
        except Exception:
            if checkpoint_disposition != "unknown":
                raise
            logger.exception("graph_run.checkpoint_cleanup_failed")
        finally:
            try:
                await run.aclose()
            except Exception:
                if checkpoint_disposition != "unknown":
                    raise
                logger.exception("graph_run.lease_release_failed")


async def delete_checkpoint_thread(run: GraphRun) -> None:
    """Delete terminal state retained only to support an active interrupt."""
    if run.checkpoint_thread_id is None:
        msg = "Interrupt-enabled run has no checkpoint thread id."
        raise RuntimeError(msg)

    checkpointer = cast("BaseCheckpointSaver", run.graph.checkpointer)
    await checkpointer.adelete_thread(run.checkpoint_thread_id)
