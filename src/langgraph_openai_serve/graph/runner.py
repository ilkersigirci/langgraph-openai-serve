"""LangGraph runner service.

This module provides functionality to run LangGraph models with an OpenAI-compatible interface.
It handles conversion between OpenAI's message format and LangChain's message format,
and provides both streaming and non-streaming interfaces for running LangGraph workflows.

Examples:
    >>> from langgraph_openai_serve.services.graph_runner import run_langgraph
    >>> response, usage = await run_langgraph("my-model", messages, registry)
    >>> from langgraph_openai_serve.services.graph_runner import run_langgraph_stream
    >>> async for chunk, metrics in run_langgraph_stream("my-model", messages, registry):
    ...     print(chunk)

The module contains the following functions:
- `convert_to_lc_messages(messages)` - Converts OpenAI messages to LangChain messages.
- `register_graphs(graphs)` - Validates and returns the provided graph dictionary.
- `run_langgraph(model, messages, graph_registry)` - Runs a LangGraph model with the given messages.
- `run_langgraph_stream(model, messages, graph_registry)` - Runs a LangGraph model in streaming mode.
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict

from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_HIDDEN

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = logging.getLogger(__name__)

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
) -> tuple[str, dict[str, int]]:
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

    # Use graph_registry.get_graph to get the graph config and then the graph
    try:
        graph_config = graph_registry.get_graph(model)
        graph = await graph_config.resolve_graph()
    except ValueError as e:
        logger.error(f"Error getting graph for model '{model}': {e}")
        raise e

    request = request or ChatCompletionRequest(model=model, messages=messages)

    # Convert OpenAI messages and adapt the request to the graph's native schemas.
    lc_messages = convert_to_lc_messages(messages)
    inputs = await graph_config.build_input(request, lc_messages)
    context = await graph_config.build_context(request)
    runnable_config = _build_runnable_config(graph_config.runtime_callbacks)

    result = await graph.ainvoke(
        inputs,
        config=runnable_config,
        context=context,
    )
    response = await graph_config.render_output(result)

    # Calculate token usage (approximate)
    prompt_tokens = sum(len((m.content or "").split()) for m in messages)
    completion_tokens = len((response or "").split())
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    logger.info(f"LangGraph completion generated in {time.time() - start_time:.2f}s")
    return response, token_usage


async def run_langgraph_stream(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None = None,
) -> AsyncGenerator[tuple[str, dict[str, int]], None]:
    """Run a LangGraph model in streaming mode.

    Args:
        model: The name of the model (graph) to run.
        messages: A list of OpenAI-compatible messages.
        graph_registry: The registry containing the graph configurations.
        request: The complete chat completion request passed to graph adapters.

    Yields:
        A tuple containing the content chunk and token usage metrics.
    """
    logger.info(f"Starting streaming LangGraph completion for model '{model}'")

    try:
        graph_config = graph_registry.get_graph(model)
        graph = await graph_config.resolve_graph()
        streamable_node_names = graph_config.streamable_node_names
    except ValueError as e:
        logger.error(f"Error getting graph for model '{model}': {e}")
        raise e

    request = request or ChatCompletionRequest(model=model, messages=messages)
    lc_messages = convert_to_lc_messages(messages)
    inputs = await graph_config.build_input(request, lc_messages)
    context = await graph_config.build_context(request)
    runnable_config = _build_runnable_config(graph_config.runtime_callbacks)

    async for event in graph.astream(
        inputs,
        config=runnable_config,
        context=context,
        stream_mode=["messages"],
        subgraphs=True,
        version="v2",
    ):
        if event.get("type") != "messages":
            continue

        message, metadata = event["data"]
        if not isinstance(message, AIMessageChunk):
            continue
        if TAG_HIDDEN in (metadata.get("tags") or []):
            continue
        if metadata.get("langgraph_node") not in streamable_node_names:
            continue

        content = str(message.text)
        if content:
            yield content, {"tokens": 1}


def _build_runnable_config(callbacks: Callbacks) -> RunnableConfig | None:
    if settings.ENABLE_LANGFUSE is True:
        if callbacks is None:
            callbacks = [langfuse_handler]
        elif isinstance(callbacks, list):
            callbacks = [*callbacks, langfuse_handler]
        else:
            callback_manager: BaseCallbackManager = callbacks.copy()
            callback_manager.add_handler(langfuse_handler)
            callbacks = callback_manager

    return RunnableConfig(callbacks=callbacks) if callbacks else None
