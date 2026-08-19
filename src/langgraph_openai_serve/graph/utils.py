"""Prepare one isolated LangGraph execution for the OpenAI API."""

import sys
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, cast

from anyio import CancelScope
from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.api.chat.utils.interrupts import parse_resume_request
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphRegistry,
)
from langgraph_openai_serve.graph.interrupt import state as interrupt_state
from langgraph_openai_serve.integrations.langfuse import get_langfuse_callback
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = get_logger(__name__)


@dataclass
class GraphRun:
    """Context for a graph run."""

    config: GraphConfig
    graph: CompiledStateGraph
    inputs: Any
    context: Any
    runnable_config: RunnableConfig | None
    run_id: str | None
    checkpoint_thread_id: str | None = None
    should_execute: bool = True
    _lease: AbstractAsyncContextManager[None] | None = field(
        default=None,
        repr=False,
    )

    async def aclose(self) -> None:
        """Release this run's single-flight lease exactly once."""
        lease, self._lease = self._lease, None
        if lease is not None:
            await lease.__aexit__(None, None, None)


async def prepare_run(  # ruff: ignore[too-many-locals]
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None,
    *,
    checkpoint_scope: str = "default",
) -> GraphRun:
    """Prepare a graph run."""
    graph_config = graph_registry.get_graph(model)

    request = request or ChatCompletionRequest(model=model, messages=messages)
    graph = await graph_config.resolve_graph()

    if not graph_config.supports(GraphFeature.INTERRUPTS):
        lc_messages = convert_to_lc_messages(messages)
        return GraphRun(
            config=graph_config,
            graph=graph,
            inputs=await graph_config.build_input(request, lc_messages),
            context=await graph_config.build_context(request, graph),
            runnable_config=build_runnable_config(graph_config.runtime_callbacks),
            run_id=None,
        )

    resume = parse_resume_request(messages)
    requested_run_id = interrupt_state.get_run_id(request)
    run_id = interrupt_state.resolve_run_id(requested_run_id, resume)
    checkpoint_thread_id = interrupt_state.checkpoint_key(
        model,
        run_id,
        scope=interrupt_state.normalize_checkpoint_scope(checkpoint_scope),
    )
    runnable_config = build_runnable_config(
        graph_config.runtime_callbacks,
        configurable={"thread_id": checkpoint_thread_id},
    )
    if runnable_config is None:  # The configurable thread always creates one.
        msg = "Interrupt run has no runnable configuration."
        raise RuntimeError(msg)

    coordinator = graph_config.run_coordinator
    if coordinator is None:  # resolve_graph() reports this as configuration error.
        msg = "Interrupt run has no coordinator."
        raise RuntimeError(msg)
    lease = coordinator(checkpoint_thread_id)
    await lease.__aenter__()  # ruff: ignore[unnecessary-dunder-call]

    try:
        snapshot = await graph.aget_state(runnable_config, subgraphs=True)
        inputs, should_execute = await interrupt_state.prepare_interrupt_input(
            graph_config,
            graph,
            request,
            snapshot,
            resume,
        )
        context = (
            await graph_config.build_context(request, graph) if should_execute else None
        )
    except BaseException:
        error_info = sys.exc_info()
        with CancelScope(shield=True):
            try:
                await lease.__aexit__(*error_info)
            except Exception:
                logger.exception("graph_run.preparation_cleanup_failed")
        raise

    return GraphRun(
        config=graph_config,
        graph=graph,
        inputs=inputs,
        context=context,
        runnable_config=runnable_config,
        run_id=run_id,
        checkpoint_thread_id=checkpoint_thread_id,
        should_execute=should_execute,
        _lease=lease,
    )


def build_runnable_config(
    callbacks: Callbacks,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig | None:
    """Build runnable config."""
    if settings.ENABLE_LANGFUSE:
        # GraphConfig is shared across requests; add tracing without mutating its
        # callback collection or manager.
        langfuse_callback = get_langfuse_callback()
        if callbacks is None:
            callbacks = [langfuse_callback]
        elif isinstance(callbacks, list):
            callbacks = [
                *cast("list[BaseCallbackHandler]", callbacks),
                langfuse_callback,
            ]
        else:
            callbacks = callbacks.copy()
            callbacks.add_handler(langfuse_callback)

    kwargs: dict[str, Any] = {}
    if callbacks:
        kwargs["callbacks"] = callbacks
    if configurable:
        kwargs["configurable"] = configurable

    return RunnableConfig(**kwargs) if kwargs else None
