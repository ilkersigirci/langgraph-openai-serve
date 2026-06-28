"""Run LangGraph workflows behind the OpenAI-compatible chat API."""

import logging
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessageChunk
from langgraph.constants import TAG_HIDDEN

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
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


LangGraphOutput = str | LangGraphInterrupt
LangGraphStreamEvent = str | LangGraphInterrupt


async def run_langgraph(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> tuple[LangGraphOutput, dict[str, int]]:
    """Run a LangGraph model with the given messages using the compiled workflow.

    This function processes input messages through a LangGraph workflow and returns
    the generated response along with token usage information.

    Examples:
        >>> response, usage = await run_langgraph("my-model", messages, registry)
        >>> print(response)
        >>> print(usage)

    Args:
        model: The name of the model to use, which also determines which graph to use.
        messages: A list of messages to process through the LangGraph.
        graph_registry: The GraphRegistry instance containing registered graphs.
        request: The complete chat completion request passed to graph adapters.

    Returns:
        A tuple containing the generated response string and a dictionary of token usage information.
    """
    logger.info(f"Running LangGraph model {model} with {len(messages)} messages")
    start_time = time.time()

    run = await prepare_run(model, messages, graph_registry, request)

    response = await invoke_run(run)

    token_usage = usage_for(response, messages)

    logger.info(f"LangGraph completion generated in {time.time() - start_time:.2f}s")
    return response, token_usage


async def invoke_run(run: GraphRun) -> LangGraphOutput:
    """Run an already prepared LangGraph invocation."""
    result = await run.graph.ainvoke(
        run.inputs,
        config=run.runnable_config,
        context=run.context,
    )
    if run.config.interrupts_enabled:
        interrupt = extract_interrupt(result, run.thread_id)
        if interrupt is not None:
            return interrupt

    return await run.config.render_output(result)


async def run_langgraph_stream(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """Run a LangGraph model in streaming mode.

    Args:
        model: The name of the model (graph) to run.
        messages: A list of OpenAI-compatible messages.
        graph_registry: The registry containing the graph configurations.
        request: The complete chat completion request passed to graph adapters.

    Yields:
        Assistant text chunks or LangGraph interrupts.
    """
    logger.info(f"Starting streaming LangGraph completion for model '{model}'")

    run = await prepare_run(model, messages, graph_registry, request)
    async for event in stream_run(run):
        yield event


async def stream_run(
    run: GraphRun,
) -> AsyncGenerator[LangGraphStreamEvent, None]:
    """Stream an already prepared LangGraph invocation."""
    stream_mode = ["messages"]
    if run.config.interrupts_enabled:
        stream_mode.append("updates")

    graph_stream = run.graph.astream(
        run.inputs,
        config=run.runnable_config,
        context=run.context,
        stream_mode=stream_mode,
        subgraphs=True,
        version="v2",
    )
    async with aclosing(graph_stream):
        async for event in graph_stream:
            if event.get("type") == "updates":
                interrupt = extract_interrupt(event.get("data"), run.thread_id)
                if interrupt is not None:
                    yield interrupt
                continue

            if event.get("type") != "messages":
                continue

            message, metadata = event["data"]
            if not isinstance(message, AIMessageChunk):
                continue
            if TAG_HIDDEN in (metadata.get("tags") or []):
                continue
            if metadata.get("langgraph_node") not in run.config.streamable_node_names:
                continue

            content = str(message.text)
            if content:
                yield content


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

    interrupt = interrupts[0]
    interrupt_id = getattr(interrupt, "id", None)
    payload = getattr(interrupt, "value", interrupt)

    if not interrupt_id:
        return None

    return LangGraphInterrupt(
        thread_id=thread_id or "",
        interrupt_id=interrupt_id,
        payload=payload,
    )
