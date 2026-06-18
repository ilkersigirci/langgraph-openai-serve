"""LangGraph runner service.

This module provides functionality to run LangGraph models with an OpenAI-compatible interface.
It handles conversion between OpenAI's message format and LangChain's message format,
and provides both streaming and non-streaming interfaces for running LangGraph workflows.

Examples:
    >>> result = await run_langgraph("my-model", messages, registry)
    >>> if result.tool_call is None:
    ...     print(result.content)
    >>> async for event in run_langgraph_stream("my-model", messages, registry):
    ...     print(event)

The module contains the following functions:
- `register_graphs(graphs)` - Validates and returns the provided graph dictionary.
- `run_langgraph(...)` - Runs a graph and returns text or a resumable interrupt.
- `run_langgraph_stream(...)` - Streams text deltas and resumable interrupts.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict

from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_HIDDEN
from langgraph.types import Command
from pydantic import BaseModel

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.hitl.openai import (
    HitlToolCall,
    create_tool_call,
    parse_resume_message,
)
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangGraphRunResult:
    """The completed or interrupted result of a graph invocation."""

    content: str
    token_usage: dict[str, int]
    tool_call: HitlToolCall | None = None


if settings.ENABLE_LANGFUSE is True:
    from langfuse.langchain import CallbackHandler

    langfuse_handler = CallbackHandler()


def register_graphs(graphs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return the provided graph dictionary.

    Args:
        graphs: A dictionary mapping graph names to LangGraph instances.

    Returns:
        The validated graph dictionary.
    """
    # Potential future validation can go here
    logger.info(f"Registered {len(graphs)} graphs: {', '.join(graphs.keys())}")
    return graphs


async def run_langgraph(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> LangGraphRunResult:
    """Run a graph and return either final text or a resumable interrupt."""
    logger.info(f"Running LangGraph model {model} with {len(messages)} messages")
    start_time = time.time()

    (
        graph_config,
        graph,
        inputs,
        context,
        runnable_config,
        thread_id,
    ) = await _prepare_run(model, messages, graph_registry, request)

    result = await graph.ainvoke(
        inputs,
        config=runnable_config,
        context=context,
        version="v2",
    )
    if result.interrupts:
        return LangGraphRunResult(
            content="",
            token_usage=_token_usage(messages, ""),
            tool_call=create_tool_call(thread_id, result.interrupts),
        )

    response = await graph_config.render_output(_graph_output(result))

    # Calculate token usage (approximate)
    token_usage = _token_usage(messages, response)

    logger.info(f"LangGraph completion generated in {time.time() - start_time:.2f}s")
    return LangGraphRunResult(content=response, token_usage=token_usage)


def _token_usage(
    messages: list[ChatCompletionRequestMessage], response: str
) -> dict[str, int]:
    prompt_tokens = sum(len((message.content or "").split()) for message in messages)
    completion_tokens = len((response or "").split())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def run_langgraph_stream(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> AsyncGenerator[str | HitlToolCall, None]:
    """Stream text deltas and resumable LangGraph interrupts."""
    logger.info(f"Starting streaming LangGraph completion for model '{model}'")

    (
        graph_config,
        graph,
        inputs,
        context,
        runnable_config,
        thread_id,
    ) = await _prepare_run(model, messages, graph_registry, request)
    streamable_node_names = graph_config.streamable_node_names
    emitted_text = False
    final_output: Any = None
    interrupts: dict[str, Any] = {}

    async for event in graph.astream(
        inputs,
        config=runnable_config,
        context=context,
        stream_mode=["messages", "values"],
        subgraphs=True,
        version="v2",
    ):
        if event.get("type") == "values" and not event.get("ns"):
            final_output = event["data"]
            for interrupt in event.get("interrupts", ()):
                interrupts[str(interrupt.id)] = interrupt
            continue

        content = _stream_content(event, streamable_node_names)
        if content:
            emitted_text = True
            yield content

    if interrupts:
        yield create_tool_call(thread_id, interrupts.values())
        return

    if not emitted_text and final_output is not None:
        content = await graph_config.render_output(_normalize_output(final_output))
        if content:
            yield content


def _stream_content(event: dict[str, Any], streamable_node_names: list[str]) -> str:
    if event.get("type") != "messages":
        return ""

    message, metadata = event["data"]
    if not isinstance(message, AIMessageChunk):
        return ""
    if TAG_HIDDEN in (metadata.get("tags") or []):
        return ""
    if metadata.get("langgraph_node") not in streamable_node_names:
        return ""
    return str(message.text)


def _graph_output(result: Any) -> Any:
    return _normalize_output(result.value if hasattr(result, "value") else result)


def _normalize_output(output: Any) -> Any:
    return output.model_dump() if isinstance(output, BaseModel) else output


async def _prepare_run(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None,
) -> tuple[Any, Any, Any, Any, RunnableConfig | None, str]:
    graph_config = graph_registry.get_graph(model)
    graph = await graph_config.resolve_graph()
    request = request or ChatCompletionRequest(model=model, messages=messages)

    resume = parse_resume_message(messages)
    if resume is None:
        thread_id = uuid.uuid4().hex
        inputs = await graph_config.build_input(
            request, convert_to_lc_messages(messages)
        )
    else:
        thread_id = resume.thread_id
        inputs = Command(resume=resume.value)

    context = await graph_config.build_context(request)
    config = _build_runnable_config(graph_config.runtime_callbacks, thread_id=thread_id)
    return graph_config, graph, inputs, context, config, thread_id


def _build_runnable_config(
    callbacks: Callbacks, thread_id: str | None = None
) -> RunnableConfig | None:
    if settings.ENABLE_LANGFUSE is True:
        if callbacks is None:
            callbacks = [langfuse_handler]
        elif isinstance(callbacks, list):
            callbacks = [*callbacks, langfuse_handler]
        else:
            callback_manager: BaseCallbackManager = callbacks.copy()
            callback_manager.add_handler(langfuse_handler)
            callbacks = callback_manager

    config = RunnableConfig()
    if callbacks:
        config["callbacks"] = callbacks
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    return config or None
