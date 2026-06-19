import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = logging.getLogger(__name__)

if settings.ENABLE_LANGFUSE is True:
    from langfuse.langchain import CallbackHandler

    langfuse_handler = CallbackHandler()


@dataclass
class GraphRun:
    config: GraphConfig
    graph: CompiledStateGraph
    inputs: Any
    context: Any
    runnable_config: RunnableConfig | None


async def prepare_run(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None,
) -> GraphRun:
    try:
        graph_config = graph_registry.get_graph(model)
        graph = await graph_config.resolve_graph()
    except ValueError as e:
        logger.error(f"Error getting graph for model '{model}': {e}")
        raise e

    request = request or ChatCompletionRequest(model=model, messages=messages)
    lc_messages = convert_to_lc_messages(messages)

    return GraphRun(
        config=graph_config,
        graph=graph,
        inputs=await graph_config.build_input(request, lc_messages),
        context=await graph_config.build_context(request),
        runnable_config=build_runnable_config(graph_config.runtime_callbacks),
    )


def build_runnable_config(callbacks: Callbacks) -> RunnableConfig | None:
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
