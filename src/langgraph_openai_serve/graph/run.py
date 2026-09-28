"""Prepare one isolated LangGraph execution for the OpenAI API."""

import uuid
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast

from anyio import fail_after
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
from langchain_core.messages import BaseMessage, UsageMetadata
from langchain_core.messages.ai import add_usage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langgraph_openai_serve.core.errors import GraphError
from langgraph_openai_serve.core.logging import (
    bind_log_context,
    get_log_context,
    get_logger,
)
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.interrupt import (
    InterruptResume,
    RunCoordinator,
    checkpoint_key,
    resume_command,
)
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.integrations.langfuse import get_langfuse_callback
from langgraph_openai_serve.protocol import CONVERSATION_METADATA_KEY

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = get_logger(__name__)

_RUN_NAME = "lgos.graph_run"
# Seconds. Cleanup is one checkpoint delete and one lease release, so only an
# unhealthy store takes this long.
_CLEANUP_TIMEOUT = 10


@dataclass(frozen=True)
class InterruptRun:
    """The durable identity of one interrupt-enabled run."""

    run_id: str
    thread_id: str


@dataclass
class GraphRun:
    """
    One prepared graph run and the resources it holds until closed.

    An interrupt-enabled run holds its coordinator lease from preparation until
    ``aclose()``. Closing deletes the checkpoint thread of a run that executed
    without pausing, so durable state exists only for pending interrupts.
    """

    config: GraphConfig
    graph: CompiledStateGraph
    inputs: Any
    context: Any
    runnable_config: RunnableConfig
    usage_callback: UsageMetadataCallbackHandler
    interrupt: InterruptRun | None = None
    _resources: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)
    _delete_checkpoint: bool = field(default=False, init=False, repr=False)

    async def __aenter__(self) -> Self:
        """Return this run; exiting closes it."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close this run."""
        await self.aclose(error=exc)

    async def hold(self, lease: AbstractAsyncContextManager[None]) -> None:
        """Hold ``lease`` until this run closes."""
        await self._resources.enter_async_context(lease)

    def begin_execution(self) -> None:
        """Mark checkpoint state for deletion unless the run later pauses."""
        self._delete_checkpoint = self.interrupt is not None

    def keep_checkpoint(self) -> None:
        """Preserve the checkpoint of a run that paused on interrupts."""
        self._delete_checkpoint = False

    async def aclose(self, *, error: BaseException | None = None) -> None:
        """
        Apply checkpoint cleanup and release the lease; later calls do nothing.

        When ``error`` ended the run, a cleanup failure is logged instead of
        replacing it.
        """
        try:
            await self._cleanup()
        except Exception:
            if error is None:
                raise
            logger.exception("graph_run.cleanup_failed")

    async def _cleanup(self) -> None:
        # Shielded because cleanup may run inside a cancelled request scope, and
        # bounded so a hung store cannot hold the request forever. Abandoning is
        # safe: a coordinator must release a cancelled lease, and an undeleted
        # checkpoint is only orphaned.
        with fail_after(_CLEANUP_TIMEOUT, shield=True):
            async with self._resources:
                if self._delete_checkpoint and self.interrupt is not None:
                    self._delete_checkpoint = False
                    checkpointer = cast("BaseCheckpointSaver", self.graph.checkpointer)
                    await checkpointer.adelete_thread(self.interrupt.thread_id)

    def usage_metadata(self) -> UsageMetadata | None:
        """Return provider-reported usage aggregated across the graph run."""
        total = None
        for usage in self.usage_callback.usage_metadata.values():
            total = add_usage(total, usage)
        return total


async def prepare_run(  # ruff: ignore[too-many-arguments] - Background runs choose their run_id before execution.
    request: GraphRequest,
    messages: list[BaseMessage],
    graph_registry: GraphRegistry,
    *,
    resume: InterruptResume | None = None,
    run_id: str | None = None,
    checkpoint_scope: str = "default",
) -> GraphRun:
    """
    Resolve, lease, and build the inputs of one graph run.

    An interrupt-enabled run continues ``resume.run_id``, uses a server-chosen
    ``run_id``, or starts a new run.
    """
    config = graph_registry.get_graph(request.model)
    graph = await config.resolve_graph()
    # A server-chosen run without a resume is a background job. When its worker
    # dies, the engine runs the job again, and it starts over.
    restart = resume is None and run_id is not None
    interrupt_run = None
    if config.supports(GraphFeature.INTERRUPTS):
        if resume is not None:
            run_id = resume.run_id
        run_id = run_id or str(uuid.uuid4())
        bind_log_context(operation_id=run_id)
        interrupt_run = InterruptRun(
            run_id=run_id,
            thread_id=checkpoint_key(request.model, run_id, scope=checkpoint_scope),
        )

    usage_callback = UsageMetadataCallbackHandler()
    runnable_config = _runnable_config(
        request,
        config.runtime_callbacks,
        usage_callback,
        interrupt_run,
    )
    run = GraphRun(
        config=config,
        graph=graph,
        inputs=None,
        context=None,
        runnable_config=runnable_config,
        usage_callback=usage_callback,
        interrupt=interrupt_run,
    )
    try:
        await _prepare_inputs(
            run, request, messages, resume, graph_registry.run_coordinator
        )
        if restart and run.interrupt is not None:
            checkpointer = cast("BaseCheckpointSaver", run.graph.checkpointer)
            if await checkpointer.aget_tuple(run.runnable_config) is not None:
                await checkpointer.adelete_thread(run.interrupt.thread_id)
    except BaseException as exc:
        await run.aclose(error=exc)
        raise
    return run


async def _prepare_inputs(
    run: GraphRun,
    request: GraphRequest,
    messages: list[BaseMessage],
    resume: InterruptResume | None,
    coordinator: RunCoordinator | None,
) -> None:
    """Lease an interrupt run, then build its native input and context."""
    if run.interrupt is not None:
        if coordinator is None:
            msg = "Interrupt-enabled graphs need a GraphRegistry run_coordinator."
            raise GraphError(msg)
        await run.hold(coordinator(run.interrupt.thread_id))
    if resume is not None and run.interrupt is not None:
        run.inputs = await resume_command(run.graph, run.runnable_config, resume)
    else:
        run.inputs = await run.config.build_input(request, messages)
    run.context = await run.config.build_context(request, run.graph)


def _runnable_config(
    request: GraphRequest,
    callbacks: Callbacks,
    usage_callback: UsageMetadataCallbackHandler,
    interrupt_run: InterruptRun | None,
) -> RunnableConfig:
    # GraphConfig is shared across requests; add request handlers without
    # mutating its callback collection or manager.
    handlers: list[BaseCallbackHandler] = [usage_callback]
    if settings.ENABLE_LANGFUSE:
        handlers.append(get_langfuse_callback())
    if callbacks is None:
        run_callbacks: Callbacks = handlers
    elif isinstance(callbacks, list):
        run_callbacks = [*callbacks, *handlers]
    else:
        run_callbacks = callbacks.copy()
        for handler in handlers:
            run_callbacks.add_handler(handler)

    metadata = {"lgos.model": request.model}
    if conversation_id := request.metadata.get(CONVERSATION_METADATA_KEY):
        metadata["langfuse_session_id"] = conversation_id
    if isinstance(request_id := get_log_context().get("request_id"), str):
        metadata["lgos.request_id"] = request_id
    if interrupt_run is None:
        return RunnableConfig(
            callbacks=run_callbacks, run_name=_RUN_NAME, metadata=metadata
        )

    metadata["lgos.operation_id"] = interrupt_run.run_id
    return RunnableConfig(
        callbacks=run_callbacks,
        run_name=_RUN_NAME,
        metadata=metadata,
        configurable={"thread_id": interrupt_run.thread_id},
    )
