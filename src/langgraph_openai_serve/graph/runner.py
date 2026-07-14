"""Run LangGraph workflows behind the OpenAI-compatible chat API."""

import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, AsyncGenerator, cast

from langchain_core.messages import AIMessageChunk
from langgraph.constants import TAG_HIDDEN
from langgraph.types import CustomStreamPart, StreamMode
from pydantic import BaseModel

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.utils import (
    GraphRun,
    prepare_run,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangGraphInterrupt:
    thread_id: str
    interrupt_id: str
    payload: Any


@dataclass(frozen=True)
class LangGraphInvocation:
    """A graph result together with custom events emitted during its run."""

    output: "LangGraphOutput"
    custom_events: tuple[CustomStreamPart, ...]


LangGraphOutput = str | LangGraphInterrupt
LangGraphStreamEvent = str | LangGraphInterrupt | CustomStreamPart

_MISSING = object()


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
            subgraphs=True,
            version="v2",
        ),
    )
    async with aclosing(graph_stream):
        async for event in graph_stream:
            if event.get("type") == "custom":
                custom_events.append(cast(CustomStreamPart, event))
                continue

            if event.get("type") == "values" and not event.get("ns"):
                if run.config.supports(GraphFeature.INTERRUPTS):
                    interrupt = extract_stream_interrupt(event, run.thread_id)
                    if interrupt is not None:
                        return LangGraphInvocation(
                            output=interrupt,
                            custom_events=tuple(custom_events),
                        )
                final_output = event.get("data")

    if final_output is _MISSING:
        raise RuntimeError("LangGraph invocation completed without a final value.")

    return LangGraphInvocation(
        output=await run.config.render_output(legacy_output(final_output)),
        custom_events=tuple(custom_events),
    )


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
    async for event in stream_run(run):
        yield event


async def stream_run(
    run: GraphRun,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """Stream an already prepared LangGraph invocation."""
    stream_mode: list[StreamMode] = ["messages", "custom"]
    if run.config.supports(GraphFeature.INTERRUPTS):
        stream_mode.append("updates")

    graph_stream = cast(
        AsyncGenerator[dict[str, Any], None],
        run.graph.astream(
            run.inputs,
            config=run.runnable_config,
            context=run.context,
            stream_mode=stream_mode,
            subgraphs=True,
            version="v2",
        ),
    )
    async with aclosing(graph_stream):
        async for event in graph_stream:
            if event.get("type") == "custom":
                yield cast(CustomStreamPart, event)
                continue

            if event.get("type") == "updates":
                interrupt = extract_interrupt(event.get("data"), run.thread_id)
                if interrupt is not None:
                    yield interrupt
                continue

            if event.get("type") != "messages":
                continue

            content = text_from_message_event(event, run)
            if content:
                yield content


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


def extract_interrupt(output: Any, thread_id: str | None) -> LangGraphInterrupt | None:
    if not isinstance(output, dict) or "__interrupt__" not in output:
        return None

    interrupts = output["__interrupt__"]
    if not interrupts:
        return None

    return interrupt_from_value(interrupts[0], thread_id)


def extract_stream_interrupt(
    event: dict,
    thread_id: str | None,
) -> LangGraphInterrupt | None:
    """Extract interrupt metadata from a LangGraph v2 values stream part."""
    interrupts = event.get("interrupts") or ()
    if not interrupts:
        return None
    return interrupt_from_value(interrupts[0], thread_id)


def interrupt_from_value(
    interrupt: Any,
    thread_id: str | None,
) -> LangGraphInterrupt | None:
    interrupt_id = getattr(interrupt, "id", None)
    payload = getattr(interrupt, "value", interrupt)

    if not interrupt_id:
        return None

    return LangGraphInterrupt(
        thread_id=thread_id or "",
        interrupt_id=interrupt_id,
        payload=payload,
    )
