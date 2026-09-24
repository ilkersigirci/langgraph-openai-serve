"""Prepare one isolated LangGraph execution for the OpenAI API."""

import sys
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from anyio import CancelScope
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
from langchain_core.messages import BaseMessage, UsageMetadata
from langchain_core.messages.ai import add_usage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

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
from langgraph_openai_serve.graph.interrupt.models import (
    InterruptResume,
    LangGraphInterruptBatch,
)
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.integrations.langfuse import get_langfuse_callback
from langgraph_openai_serve.protocol import CONVERSATION_METADATA_KEY

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = get_logger(__name__)
_RUN_NAME = "lgos.graph_run"
_LANGFUSE_SESSION_ID_METADATA_KEY = "langfuse_session_id"
_CheckpointDisposition = Literal["untouched", "delete", "preserve"]


@dataclass(frozen=True)
class _RunIdentity:
    run_id: str | None = None
    checkpoint_thread_id: str | None = None

    def configurable(self) -> dict[str, str] | None:
        if self.checkpoint_thread_id is None:
            return None
        return {"thread_id": self.checkpoint_thread_id}


@dataclass(frozen=True)
class _PreparedRunValues:
    inputs: Any
    context: Any
    pending_batch: LangGraphInterruptBatch | None = None


@dataclass
class GraphRun:
    """Own one prepared graph run and its cleanup resources."""

    config: GraphConfig
    graph: CompiledStateGraph
    inputs: Any
    context: Any
    runnable_config: RunnableConfig | None
    run_id: str | None
    checkpoint_thread_id: str | None = None
    pending_batch: LangGraphInterruptBatch | None = None
    usage_callback: UsageMetadataCallbackHandler = field(
        default_factory=UsageMetadataCallbackHandler,
        repr=False,
    )
    _resources: AsyncExitStack = field(
        default_factory=AsyncExitStack,
        repr=False,
    )
    _checkpoint_disposition: _CheckpointDisposition = field(
        default="untouched",
        init=False,
        repr=False,
    )
    _primary_error: BaseException | None = field(default=None, init=False, repr=False)
    _entered: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    async def __aenter__(self) -> Self:
        """Claim ownership of this prepared run."""
        if self._closed:
            msg = "A closed graph run cannot be reused."
            raise RuntimeError(msg)
        if self._entered:
            msg = "A graph run can have only one active owner."
            raise RuntimeError(msg)
        self._entered = True
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Finalize this run without suppressing its primary failure."""
        if exc is not None:
            self.record_failure(exc)
        try:
            await self.aclose()
        finally:
            self._entered = False

    def begin_execution(self) -> None:
        """Mark checkpoint state as incomplete immediately before execution."""
        self.require_owner()
        if self.config.supports(GraphFeature.INTERRUPTS):
            self._checkpoint_disposition = "delete"

    def commit_interrupts(self) -> None:
        """Preserve a validated interrupt batch committed by the runner."""
        self.require_owner()
        if not self.config.supports(GraphFeature.INTERRUPTS):
            msg = "Only interrupt-enabled runs can commit an interrupt batch."
            raise RuntimeError(msg)
        self._checkpoint_disposition = "preserve"

    def record_failure(self, error: BaseException) -> None:
        """Retain the first failure so later cleanup cannot replace it."""
        if self._primary_error is None:
            self._primary_error = error

    async def aclose(self) -> None:
        """Apply checkpoint disposition and release resources exactly once."""
        if self._closed:
            return
        self._closed = True

        with CancelScope(shield=True):
            cleanup_error: BaseException | None = None
            if self._checkpoint_disposition == "delete":
                try:
                    await self._delete_checkpoint_thread()
                except BaseException as exc:
                    if self._primary_error is None:
                        cleanup_error = exc
                    else:
                        logger.exception("graph_run.checkpoint_cleanup_failed")

            active_error = self._primary_error or cleanup_error
            try:
                await self._resources.__aexit__(
                    type(active_error) if active_error is not None else None,
                    active_error,
                    active_error.__traceback__ if active_error is not None else None,
                )
            except BaseException:
                if active_error is None:
                    raise
                logger.exception("graph_run.lease_release_failed")

            if cleanup_error is not None:
                raise cleanup_error

    def require_owner(self) -> None:
        """Require the caller to own this run through its async context."""
        if not self._entered:
            msg = "Graph execution requires an active GraphRun context."
            raise RuntimeError(msg)

    async def _delete_checkpoint_thread(self) -> None:
        if self.checkpoint_thread_id is None:
            msg = "Interrupt-enabled run has no checkpoint thread id."
            raise RuntimeError(msg)

        checkpointer = cast("BaseCheckpointSaver", self.graph.checkpointer)
        await checkpointer.adelete_thread(self.checkpoint_thread_id)

    def usage_metadata(self) -> UsageMetadata | None:
        """Return provider-reported usage aggregated across the graph run."""
        total = None
        for usage in self.usage_callback.usage_metadata.values():
            total = add_usage(total, usage)
        return total


async def prepare_run(
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_registry: GraphRegistry,
    *,
    resume: InterruptResume | None = None,
    checkpoint_scope: str = "default",
) -> GraphRun:
    """Prepare a graph run."""
    graph_config = graph_registry.get_graph(request.model)
    graph = await graph_config.resolve_graph()
    usage_callback = UsageMetadataCallbackHandler()
    identity = _resolve_run_identity(
        request,
        graph_config,
        resume,
        checkpoint_scope=checkpoint_scope,
    )
    runnable_config = build_runnable_config(
        graph_config.runtime_callbacks,
        configurable=identity.configurable(),
        metadata=_runnable_metadata(request, identity.run_id),
        extra_callbacks=[usage_callback],
    )
    if identity.checkpoint_thread_id is not None and runnable_config is None:
        msg = "Interrupt run has no runnable configuration."
        raise RuntimeError(msg)

    resources = AsyncExitStack()
    try:
        values = await _prepare_run_values(
            request=request,
            messages=messages,
            graph_config=graph_config,
            graph=graph,
            runnable_config=runnable_config,
            identity=identity,
            resources=resources,
            resume=resume,
        )
    except BaseException:
        error_info = sys.exc_info()
        with CancelScope(shield=True):
            try:
                await resources.__aexit__(*error_info)
            except BaseException:
                logger.exception("graph_run.preparation_cleanup_failed")
        raise

    return GraphRun(
        config=graph_config,
        graph=graph,
        inputs=values.inputs,
        context=values.context,
        runnable_config=runnable_config,
        run_id=identity.run_id,
        checkpoint_thread_id=identity.checkpoint_thread_id,
        pending_batch=values.pending_batch,
        usage_callback=usage_callback,
        _resources=resources,
    )


def _resolve_run_identity(
    request: GraphRequest,
    graph_config: GraphConfig,
    resume: InterruptResume | None,
    *,
    checkpoint_scope: str,
) -> _RunIdentity:
    if not graph_config.supports(GraphFeature.INTERRUPTS):
        return _RunIdentity()

    requested_run_id = interrupt_state.get_run_id(request)
    run_id = interrupt_state.resolve_run_id(requested_run_id, resume)
    bind_log_context(operation_id=run_id)
    checkpoint_thread_id = interrupt_state.checkpoint_key(
        request.model,
        run_id,
        scope=interrupt_state.normalize_checkpoint_scope(checkpoint_scope),
    )
    return _RunIdentity(
        run_id=run_id,
        checkpoint_thread_id=checkpoint_thread_id,
    )


async def _prepare_run_values(  # ruff: ignore[too-many-arguments] - One resource boundary needs every preparation input.
    *,
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_config: GraphConfig,
    graph: CompiledStateGraph,
    runnable_config: RunnableConfig | None,
    identity: _RunIdentity,
    resources: AsyncExitStack,
    resume: InterruptResume | None,
) -> _PreparedRunValues:
    if identity.checkpoint_thread_id is None:
        inputs = await graph_config.build_input(request, messages)
        context = await graph_config.build_context(request, graph)
        return _PreparedRunValues(inputs=inputs, context=context)

    coordinator = graph_config.run_coordinator
    if coordinator is None:  # resolve_graph() reports this first.
        msg = "Interrupt run has no coordinator."
        raise RuntimeError(msg)
    if runnable_config is None or identity.run_id is None:
        msg = "Interrupt run has no runnable configuration."
        raise RuntimeError(msg)

    await resources.enter_async_context(coordinator(identity.checkpoint_thread_id))
    state = await interrupt_state.prepare_interrupt_state(
        graph,
        runnable_config,
        identity.run_id,
        resume,
    )
    if isinstance(state, LangGraphInterruptBatch):
        return _PreparedRunValues(inputs=None, context=None, pending_batch=state)
    return _PreparedRunValues(
        inputs=(
            state
            if state is not None
            else await graph_config.build_input(request, messages)
        ),
        context=await graph_config.build_context(request, graph),
    )


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
    request: GraphRequest,
    run_id: str | None = None,
) -> dict[str, str]:
    """Build correlation metadata for callbacks and tracing."""
    metadata = {
        "lgos.model": request.model,
    }
    conversation_id = request.metadata.get(CONVERSATION_METADATA_KEY)
    if conversation_id:
        metadata[_LANGFUSE_SESSION_ID_METADATA_KEY] = conversation_id
    request_id = get_log_context().get("request_id")
    if isinstance(request_id, str):
        metadata["lgos.request_id"] = request_id
    if run_id is not None:
        metadata["lgos.operation_id"] = run_id
    return metadata
