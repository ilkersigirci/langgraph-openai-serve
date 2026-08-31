"""Prepare one isolated LangGraph execution for the OpenAI API."""

import sys
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, cast

from anyio import CancelScope
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
from langchain_core.messages import UsageMetadata
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.api.chat.utils.interrupts import parse_resume_request
from langgraph_openai_serve.core.logging import (
    bind_log_context,
    get_log_context,
    get_logger,
)
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
_RUN_NAME = "lgos.chat_completion"
_SESSION_ID_METADATA_KEY = "session_id"
_LANGFUSE_SESSION_ID_METADATA_KEY = "langfuse_session_id"


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
    usage_callback: UsageMetadataCallbackHandler = field(
        default_factory=UsageMetadataCallbackHandler,
        repr=False,
    )
    _lease: AbstractAsyncContextManager[None] | None = field(
        default=None,
        repr=False,
    )

    async def aclose(self) -> None:
        """Release this run's interrupt lease exactly once, if present."""
        lease, self._lease = self._lease, None
        if lease is not None:
            await lease.__aexit__(None, None, None)

    def usage_metadata(self) -> UsageMetadata | None:
        """Return provider-reported usage aggregated across the graph run."""
        usages = self.usage_callback.usage_metadata.values()
        if not usages:
            return None
        return UsageMetadata(
            input_tokens=sum(usage["input_tokens"] for usage in usages),
            output_tokens=sum(usage["output_tokens"] for usage in usages),
            total_tokens=sum(usage["total_tokens"] for usage in usages),
        )


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
    usage_callback = UsageMetadataCallbackHandler()

    if not graph_config.supports(GraphFeature.INTERRUPTS):
        lc_messages = convert_to_lc_messages(messages)
        inputs = await graph_config.build_input(request, lc_messages)
        context = await graph_config.build_context(request, graph)
        runnable_config = build_runnable_config(
            graph_config.runtime_callbacks,
            metadata=_runnable_metadata(request),
            extra_callbacks=[usage_callback],
        )
        return GraphRun(
            config=graph_config,
            graph=graph,
            inputs=inputs,
            context=context,
            runnable_config=runnable_config,
            run_id=None,
            usage_callback=usage_callback,
        )

    resume = parse_resume_request(messages)
    requested_run_id = interrupt_state.get_run_id(request)
    run_id = interrupt_state.resolve_run_id(requested_run_id, resume)
    bind_log_context(operation_id=run_id)
    checkpoint_thread_id = interrupt_state.checkpoint_key(
        model,
        run_id,
        scope=interrupt_state.normalize_checkpoint_scope(checkpoint_scope),
    )
    runnable_config = build_runnable_config(
        graph_config.runtime_callbacks,
        configurable={"thread_id": checkpoint_thread_id},
        metadata=_runnable_metadata(request, run_id),
        extra_callbacks=[usage_callback],
    )
    if runnable_config is None:  # The configurable thread always creates one.
        msg = "Interrupt run has no runnable configuration."
        raise RuntimeError(msg)

    lease = await _acquire_lease(graph_config, checkpoint_thread_id)

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
        usage_callback=usage_callback,
        _lease=lease,
    )


async def _acquire_lease(
    graph_config: GraphConfig,
    key: str,
) -> AbstractAsyncContextManager[None]:
    coordinator = graph_config.run_coordinator
    if coordinator is None:  # resolve_graph() reports this as configuration error.
        msg = "Interrupt run has no coordinator."
        raise RuntimeError(msg)

    lease = coordinator(key)
    await lease.__aenter__()  # ruff: ignore[unnecessary-dunder-call]
    return lease


def build_runnable_config(
    callbacks: Callbacks,
    configurable: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    extra_callbacks: Sequence[BaseCallbackHandler] = (),
) -> RunnableConfig | None:
    """Build runnable config."""
    callbacks = _extend_callbacks(callbacks, extra_callbacks)
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
    if kwargs:
        kwargs["run_name"] = _RUN_NAME
        if metadata:
            kwargs["metadata"] = metadata

    return RunnableConfig(**kwargs) if kwargs else None


def _extend_callbacks(
    callbacks: Callbacks,
    extra_callbacks: Sequence[BaseCallbackHandler],
) -> Callbacks:
    """Add request-owned handlers without mutating registered callbacks."""
    if not extra_callbacks:
        return callbacks
    if callbacks is None:
        return list(extra_callbacks)
    if isinstance(callbacks, list):
        return [*callbacks, *extra_callbacks]

    callbacks = callbacks.copy()
    for callback in extra_callbacks:
        callbacks.add_handler(callback)
    return callbacks


def _runnable_metadata(
    request: ChatCompletionRequest,
    run_id: str | None = None,
) -> dict[str, str]:
    """Build correlation metadata for callbacks and tracing."""
    metadata = {
        "lgos.model": request.model,
    }
    session_id = (request.metadata or {}).get(_SESSION_ID_METADATA_KEY)
    if session_id:
        metadata[_LANGFUSE_SESSION_ID_METADATA_KEY] = session_id
    request_id = get_log_context().get("request_id")
    if isinstance(request_id, str):
        metadata["lgos.request_id"] = request_id
    if run_id is not None:
        metadata["lgos.operation_id"] = run_id
    return metadata
