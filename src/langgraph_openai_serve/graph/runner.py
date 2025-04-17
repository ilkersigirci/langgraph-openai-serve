"""LangGraph runner service.

This module provides functionality to run LangGraph models with an OpenAI-compatible interface.
It handles conversion between OpenAI's message format and LangChain's message format,
and provides both streaming and non-streaming interfaces for running LangGraph workflows.

Examples:
    >>> from langgraph_openai_serve.services.graph_runner import run_langgraph
    >>> response, usage = await run_langgraph("my-model", messages)
    >>> from langgraph_openai_serve.services.graph_runner import run_langgraph_stream
    >>> async for chunk, metrics in run_langgraph_stream("my-model", messages):
    ...     print(chunk)

The module contains the following functions:
- `convert_to_lc_messages(messages)` - Converts OpenAI messages to LangChain messages.
- `get_graph_registry()` - Gets the graph registry.
- `run_langgraph(model, messages, temperature, max_tokens, tools)` - Runs a LangGraph model with the given messages.
- `run_langgraph_stream(model, messages, temperature, max_tokens, tools)` - Runs a LangGraph model in streaming mode.
"""

import logging
import time
from typing import Any, Dict

from langchain_core.messages import AIMessageChunk

from langgraph_openai_serve.schemas.openai_schema import (
    ChatCompletionRequestMessage,
    Tool,
)
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = logging.getLogger(__name__)

# Global registry for storing graphs
GRAPH_REGISTRY = {}


def get_graph_registry() -> Dict[str, Any]:
    """Get the graph registry.

    Returns:
        The graph registry dictionary.
    """
    return GRAPH_REGISTRY


def register_graphs(graphs: Dict[str, Any]) -> None:
    """Register LangGraph instances in the global registry.

    Args:
        graphs: A dictionary mapping graph names to LangGraph instances.
    """
    GRAPH_REGISTRY.update(graphs)

    logger.info(f"Registered {len(graphs)} graphs: {', '.join(graphs.keys())}")


def get_graph_for_model(model_name: str) -> Any:
    """Get the graph for a given model name from the registry.

    The model name must exactly match a registered graph name.

    Args:
        model_name: The name of the model to get the graph for.

    Returns:
        The appropriate LangGraph instance.

    Raises:
        ValueError: If no graphs are registered or if the model name does not match any registered graph.
    """
    if not GRAPH_REGISTRY:
        raise ValueError("No graphs registered. Please register graphs before running.")

    # Check if the model name matches a registered graph name
    if model_name in GRAPH_REGISTRY:
        logger.info(f"Using graph '{model_name}' for model '{model_name}'")
        return GRAPH_REGISTRY[model_name]

    # If we get here, the model name doesn't match any registered graph
    available_models = ", ".join(GRAPH_REGISTRY.keys())
    raise ValueError(
        f"Model '{model_name}' not found in registry. Available models: {available_models}"
    )


async def run_langgraph(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    tools: list[Tool] | None = None,
) -> tuple[str, dict[str, int]]:
    """Run a LangGraph model with the given messages using the compiled workflow.

    This function processes input messages through a LangGraph workflow and returns
    the generated response along with token usage information.

    Examples:
        >>> response, usage = await run_langgraph("my-model", messages)
        >>> print(response)
        >>> print(usage)

    Args:
        model: The name of the model to use, which also determines which graph to use.
        messages: A list of messages to process through the LangGraph.
        temperature: Optional; The temperature to use for generation. Defaults to 0.7.
        max_tokens: Optional; The maximum number of tokens to generate. Defaults to None.
        tools: Optional; A list of tools available to the model. Defaults to None.

    Returns:
        A tuple containing the generated response string and a dictionary of token usage information.
    """
    logger.info(f"Running LangGraph model {model} with {len(messages)} messages")
    start_time = time.time()

    graph = get_graph_for_model(model)

    # Convert OpenAI messages to LangChain messages
    lc_messages = convert_to_lc_messages(messages)

    # Run the graph with the messages
    result = await graph.ainvoke({"messages": lc_messages})
    response = result["messages"][-1].content if result["messages"] else ""

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
    temperature: float = 0.7,
    max_tokens: int | None = None,
    tools: list[Tool] | None = None,
):
    """Run a LangGraph model in streaming mode using the compiled workflow.

    This function processes input messages through a LangGraph workflow and yields
    response chunks as they become available.

    Examples:
        >>> async for chunk, metrics in run_langgraph_stream("my-model", messages):
        ...     print(chunk)

    Args:
        model: The name of the model to use, which also determines which graph to use.
        messages: A list of messages to process through the LangGraph.
        temperature: Optional; The temperature to use for generation. Defaults to 0.7.
        max_tokens: Optional; The maximum number of tokens to generate. Defaults to None.
        tools: Optional; A list of tools available to the model. Defaults to None.

    Yields:
        Tuples containing text chunks and metrics as they are generated.
    """
    logger.info(
        f"Running LangGraph model {model} in streaming mode with {len(messages)} messages"
    )

    graph = get_graph_for_model(model)

    # Convert OpenAI messages to LangChain messages
    lc_messages = convert_to_lc_messages(messages)

    # Assume all nodes in the graph that might stream are called "generate"
    # This could be made configurable in the future
    streamable_node_names = ["generate"]
    inputs = {"messages": lc_messages}

    async for event in graph.astream_events(inputs, version="v2"):
        event_kind = event["event"]
        langgraph_node = event["metadata"].get("langgraph_node", None)

        if event_kind == "on_chat_model_stream":
            if langgraph_node not in streamable_node_names:
                continue

            ai_message_chunk: AIMessageChunk = event["data"]["chunk"]
            ai_message_content = ai_message_chunk.content
            if ai_message_content:
                yield f"{ai_message_content}", {"tokens": 1}
