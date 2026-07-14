import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.api.chat.utils.interrupts import (
    NO_RESUME,
    parse_resume_value,
)
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = logging.getLogger(__name__)

THREAD_METADATA_KEY = "langgraph_thread_id"

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
    thread_id: str | None


class MissingThreadIDError(ValueError):
    """Raised when an interrupt-enabled graph request has no thread id."""


async def prepare_run(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None,
) -> GraphRun:
    try:
        graph_config = graph_registry.get_graph(model)
    except ValueError as e:
        logger.error(f"Error getting graph for model '{model}': {e}")
        raise e

    request = request or ChatCompletionRequest(model=model, messages=messages)
    thread_id, resume_value = validate_interrupt_request(
        graph_config,
        messages,
        request,
    )
    graph = await graph_config.resolve_graph()

    if resume_value is NO_RESUME:
        lc_messages = convert_to_lc_messages(messages)
        inputs = await graph_config.build_input(request, lc_messages)
    else:
        inputs = Command(resume=resume_value)

    return GraphRun(
        config=graph_config,
        graph=graph,
        inputs=inputs,
        context=await graph_config.build_context(request),
        runnable_config=build_runnable_config(
            graph_config.runtime_callbacks,
            configurable={"thread_id": thread_id} if thread_id else None,
        ),
        thread_id=thread_id,
    )


def validate_interrupt_request(
    graph_config: GraphConfig,
    messages: list[ChatCompletionRequestMessage],
    request: ChatCompletionRequest,
) -> tuple[str | None, Any]:
    thread_id = get_thread_id(request)
    if graph_config.supports(GraphFeature.INTERRUPTS) and not thread_id:
        raise MissingThreadIDError(
            f"metadata.{THREAD_METADATA_KEY} is required for interrupt-enabled graphs."
        )

    resume_value = (
        parse_resume_value(messages)
        if graph_config.supports(GraphFeature.INTERRUPTS)
        else NO_RESUME
    )

    return thread_id, resume_value


def get_thread_id(request: ChatCompletionRequest) -> str | None:
    return (request.metadata or {}).get(THREAD_METADATA_KEY)


def build_runnable_config(
    callbacks: Callbacks,
    configurable: dict[str, Any] | None = None,
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

    kwargs: dict[str, Any] = {}
    if callbacks:
        kwargs["callbacks"] = callbacks
    if configurable:
        kwargs["configurable"] = configurable

    return RunnableConfig(**kwargs) if kwargs else None
