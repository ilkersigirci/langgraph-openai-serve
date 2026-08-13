"""Run LangGraph workflows behind the OpenAI-compatible chat API."""

import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, AsyncGenerator, Literal, cast

from anyio import CancelScope
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import TAG_HIDDEN
from langgraph.types import CustomStreamPart, Interrupt, StreamMode
from pydantic import BaseModel

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.api.chat.utils.interrupts import (
    validate_interrupt_payload,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    _interrupts_by_id,
    checkpoint_state_token,
    prepare_run,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangGraphInterruptBatch:
    """The durable interrupts awaiting answers for one graph run."""

    run_id: str
    state_token: str
    interrupts: tuple[Interrupt, ...]


@dataclass(frozen=True)
class LangGraphInvocation:
    """A graph result together with custom events emitted during its run."""

    output: "LangGraphOutput"
    custom_events: tuple[CustomStreamPart, ...]


LangGraphOutput = str | LangGraphInterruptBatch
LangGraphStreamEvent = str | LangGraphInterruptBatch | CustomStreamPart

_MISSING = object()
_CheckpointDisposition = Literal["unknown", "preserve", "delete"]


async def run_langgraph(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> LangGraphInvocation:
    """Prepare and invoke a graph for direct runner callers.

    This convenience wrapper combines :func:`prepare_run` and :func:`invoke_run`.
    The HTTP route prepares its run before creating a response so preparation
    errors can be returned as OpenAI-compatible HTTP errors; its service therefore
    calls ``invoke_run`` directly with that prepared run.

    Examples:
        >>> invocation = await run_langgraph("my-model", messages, registry)
        >>> print(invocation.output)
        >>> print(invocation.custom_events)

    Args:
        model: The name of the model to use, which also determines which graph to use.
        messages: A list of messages to process through the LangGraph.
        graph_registry: The GraphRegistry instance containing registered graphs.
        request: The complete chat completion request passed to graph adapters.

    Returns:
        The graph output and custom events emitted during the invocation.
    """
    logger.info(f"Running LangGraph model {model} with {len(messages)} messages")
    start_time = time.time()

    run = await prepare_run(model, messages, graph_registry, request)

    invocation = await invoke_run(run)

    logger.info(f"LangGraph completion generated in {time.time() - start_time:.2f}s")
    return invocation


async def invoke_run(run: GraphRun) -> LangGraphInvocation:
    """Invoke a graph and collect its custom events."""
    checkpoint_disposition: _CheckpointDisposition = "unknown"
    try:
        if not run.should_execute:
            interrupt_batch = await durable_interrupt_batch(run)
            if interrupt_batch is None:
                raise RuntimeError("Pending interrupt state disappeared before use.")
            checkpoint_disposition = "preserve"
            return LangGraphInvocation(output=interrupt_batch, custom_events=())

        stream_mode: list[StreamMode] = ["values", "custom"]

        final_output: Any = _MISSING
        custom_events: list[CustomStreamPart] = []
        graph_stream = cast(
            AsyncGenerator[dict[str, Any], None],
            run.graph.astream(
                run.inputs,
                config=run.runnable_config,
                context=run.context,
                stream_mode=stream_mode,
                output_keys=run.graph.output_channels,
                **_astream_options(run),
            ),
        )
        async with aclosing(graph_stream):
            async for event in graph_stream:
                if event.get("type") == "custom":
                    custom_events.append(cast(CustomStreamPart, event))
                    continue

                # Subgraph values share this stream, but only the root namespace is
                # the registered graph's final output.
                if event.get("type") == "values" and not event.get("ns"):
                    final_output = event.get("data")

        if run.config.supports(GraphFeature.INTERRUPTS):
            interrupt_batch = await durable_interrupt_batch(run)
            if interrupt_batch is not None:
                checkpoint_disposition = "preserve"
                return LangGraphInvocation(
                    output=interrupt_batch,
                    custom_events=tuple(custom_events),
                )

        if final_output is _MISSING:
            raise RuntimeError("LangGraph invocation completed without a final value.")

        rendered_output = await run.config.render_output(legacy_output(final_output))
        if run.config.supports(GraphFeature.INTERRUPTS):
            checkpoint_disposition = "delete"

        return LangGraphInvocation(
            output=rendered_output,
            custom_events=tuple(custom_events),
        )
    finally:
        await finalize_run(run, checkpoint_disposition)


async def run_langgraph_stream(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """Prepare and stream a graph for direct runner callers.

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
    logger.info(f"Starting streaming LangGraph completion for model '{model}'")

    run = await prepare_run(model, messages, graph_registry, request)
    run_stream = stream_run(run)
    async with aclosing(run_stream):
        async for event in run_stream:
            yield event


async def stream_run(
    run: GraphRun,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """Stream an already prepared LangGraph invocation."""
    checkpoint_disposition: _CheckpointDisposition = "unknown"
    try:
        if not run.should_execute:
            interrupt_batch = await durable_interrupt_batch(run)
            if interrupt_batch is None:
                raise RuntimeError("Pending interrupt state disappeared before use.")
            checkpoint_disposition = "preserve"
            yield interrupt_batch
            return

        stream_mode: list[StreamMode] = ["messages", "custom"]

        graph_stream = cast(
            AsyncGenerator[dict[str, Any], None],
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
                    yield cast(CustomStreamPart, event)
                    continue

                if event.get("type") != "messages":
                    continue

                content = text_from_message_event(event, run)
                if content:
                    yield content

        if run.config.supports(GraphFeature.INTERRUPTS):
            interrupt_batch = await durable_interrupt_batch(run)
            if interrupt_batch is not None:
                checkpoint_disposition = "preserve"
                yield interrupt_batch
            else:
                checkpoint_disposition = "delete"
    finally:
        await finalize_run(run, checkpoint_disposition)


def text_from_message_event(event: dict, run: GraphRun) -> str | None:
    """Extract visible text from a streamable LangGraph message event."""
    message, metadata = event["data"]
    if not isinstance(message, AIMessageChunk):
        return None
    if TAG_HIDDEN in (metadata.get("tags") or []):
        return None
    if metadata.get("langgraph_node") not in run.config.streamable_node_names:
        return None

    content = str(message.text)
    return content or None


def legacy_output(output: Any) -> Any:
    """Match the plain output shape adapters received from LangGraph v1."""
    if isinstance(output, BaseModel):
        return dict(output)
    if is_dataclass(output) and not isinstance(output, type):
        return {field.name: getattr(output, field.name) for field in fields(output)}
    return output


def _astream_options(run: GraphRun) -> dict[str, Any]:
    """Build the shared execution options for LangGraph event streams."""
    options: dict[str, Any] = {"subgraphs": True, "version": "v2"}
    if run.config.supports(GraphFeature.INTERRUPTS):
        options["durability"] = "exit"
    return options


def usage_for(
    output: LangGraphOutput,
    messages: list[ChatCompletionRequestMessage],
) -> dict[str, int]:
    prompt_tokens = sum(len((message.content or "").split()) for message in messages)
    completion_tokens = len(output.split()) if isinstance(output, str) else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def durable_interrupt_batch(run: GraphRun) -> LangGraphInterruptBatch | None:
    """Read the durable checkpoint head after graph execution has quiesced."""
    if run.runnable_config is None:
        raise RuntimeError("Interrupt-enabled runs require runnable configuration.")

    snapshot = await run.graph.aget_state(run.runnable_config, subgraphs=True)
    if not snapshot.interrupts:
        return None

    interrupts_by_id = _interrupts_by_id(snapshot)
    for interrupt in interrupts_by_id.values():
        validate_interrupt_payload(interrupt.value)

    assert run.run_id is not None
    state_token = await checkpoint_state_token(run.graph, snapshot.config)
    if state_token is None:
        raise RuntimeError("Interrupted LangGraph state has no checkpoint tuple.")
    return LangGraphInterruptBatch(
        run_id=run.run_id,
        state_token=state_token,
        interrupts=tuple(interrupts_by_id.values()),
    )


async def finalize_run(
    run: GraphRun,
    checkpoint_disposition: _CheckpointDisposition,
) -> None:
    """Apply checkpoint retention policy and release the run lease."""
    with CancelScope(shield=True):
        if checkpoint_disposition == "unknown":
            try:
                try:
                    if run.config.supports(GraphFeature.INTERRUPTS):
                        await delete_checkpoint_thread(run)
                except Exception:
                    logger.exception(
                        "Could not clean up an incomplete checkpoint thread."
                    )
            finally:
                try:
                    await run.aclose()
                except Exception:
                    logger.exception("Could not release the graph run lease.")
            return

        try:
            if checkpoint_disposition == "delete":
                await delete_checkpoint_thread(run)
        finally:
            await run.aclose()


async def delete_checkpoint_thread(run: GraphRun) -> None:
    """Delete terminal state retained only to support an active interrupt."""
    if run.checkpoint_thread_id is None:
        raise RuntimeError("Interrupt-enabled run has no checkpoint thread id.")

    checkpointer = cast(BaseCheckpointSaver, run.graph.checkpointer)
    await checkpointer.adelete_thread(run.checkpoint_thread_id)
