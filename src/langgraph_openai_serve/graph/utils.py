"""Prepare one isolated LangGraph execution for the OpenAI API."""

import hashlib
import json
import logging
import sys
import uuid
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, cast

from anyio import CancelScope
from langchain_core.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    Callbacks,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    RESUME,
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StateSnapshot

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
)
from langgraph_openai_serve.api.chat.utils.interrupts import (
    InterruptResume,
    InvalidResumeRequestError,
    parse_resume_request,
)
from langgraph_openai_serve.core.settings import settings
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
    GraphRegistry,
)
from langgraph_openai_serve.utils.message import convert_to_lc_messages

logger = logging.getLogger(__name__)

RUN_METADATA_KEY = "langgraph_run_id"

if settings.ENABLE_LANGFUSE is True:
    from langfuse.langchain import CallbackHandler

    langfuse_handler = cast(BaseCallbackHandler, CallbackHandler())


@dataclass
class GraphRun:
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


class InvalidRunIDError(ValueError):
    """Raised when a caller-supplied run id is not a UUID."""


class InterruptStateConflictError(RuntimeError):
    """Raised when a resume does not match durable pending state."""


async def prepare_run(
    model: str,
    messages: list[ChatCompletionRequestMessage],
    graph_registry: GraphRegistry,
    request: ChatCompletionRequest | None,
    *,
    checkpoint_scope: str = "default",
) -> GraphRun:
    try:
        graph_config = graph_registry.get_graph(model)
    except ValueError as exc:
        logger.error(f"Error getting graph for model '{model}': {exc}")
        raise

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
    requested_run_id = get_run_id(request)
    run_id = _resolve_run_id(requested_run_id, resume)
    checkpoint_thread_id = checkpoint_key(
        model,
        run_id,
        scope=normalize_checkpoint_scope(checkpoint_scope),
    )
    runnable_config = build_runnable_config(
        graph_config.runtime_callbacks,
        configurable={"thread_id": checkpoint_thread_id},
    )
    if runnable_config is None:  # The configurable thread always creates one.
        raise RuntimeError("Interrupt run has no runnable configuration.")

    coordinator = graph_config.run_coordinator
    if coordinator is None:  # resolve_graph() reports this as configuration error.
        raise RuntimeError("Interrupt run has no coordinator.")
    lease = coordinator(checkpoint_thread_id)
    await lease.__aenter__()

    try:
        snapshot = await graph.aget_state(runnable_config, subgraphs=True)
        inputs, should_execute = await _prepare_interrupt_input(
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
                logger.exception(
                    "Could not release a graph-run lease after preparation failed."
                )
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


async def _prepare_interrupt_input(
    graph_config: GraphConfig,
    graph: CompiledStateGraph,
    request: ChatCompletionRequest,
    snapshot: StateSnapshot,
    resume: InterruptResume | None,
) -> tuple[Any, bool]:
    pending_interrupts = _interrupts_by_id(snapshot)
    checkpoint_id = get_checkpoint_id(snapshot.config)

    if resume is None:
        if checkpoint_id is None:
            lc_messages = convert_to_lc_messages(request.messages)
            return await graph_config.build_input(request, lc_messages), True
        if pending_interrupts:
            # Re-emit persisted tool calls without rerunning graph nodes.
            return None, False
        raise InterruptStateConflictError("This run_id has already been used.")

    if checkpoint_id is None:
        raise InterruptStateConflictError(
            "No durable interrupt state exists for this run."
        )
    if not pending_interrupts:
        raise InterruptStateConflictError("This run no longer has pending interrupts.")

    checkpoint_tuple = await get_checkpoint_tuple(graph, snapshot.config)
    if checkpoint_tuple is None:
        raise InterruptStateConflictError(
            "No durable interrupt state exists for this run."
        )
    return (
        _resume_interrupt_inputs(
            checkpoint_tuple,
            set(pending_interrupts),
            resume,
        ),
        True,
    )


def _resume_interrupt_inputs(
    checkpoint_tuple: CheckpointTuple,
    pending_ids: set[str],
    resume: InterruptResume,
) -> Command:
    if resume.state_token != checkpoint_state_token(checkpoint_tuple):
        raise InterruptStateConflictError(
            "The interrupt result is stale for the current interrupt generation."
        )
    if set(resume.values) != pending_ids:
        raise InterruptStateConflictError(
            "Interrupt results do not match the complete pending interrupt set."
        )

    # Always use the ID/value form, including for one interrupt. It preserves
    # OpenAI tool_call causality and handles JSON null as a legitimate answer.
    return Command(resume=resume.values)


def _interrupts_by_id(snapshot: StateSnapshot) -> dict[str, Interrupt]:
    pending: dict[str, Interrupt] = {}
    for interrupt in snapshot.interrupts:
        interrupt_id = interrupt.id
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise RuntimeError("Durable interrupt state has an invalid interrupt id.")
        if interrupt_id in pending:
            raise RuntimeError("Durable interrupt state has duplicate interrupt ids.")
        pending[interrupt_id] = interrupt
    return pending


def _resolve_run_id(
    requested_run_id: str | None,
    resume: InterruptResume | None,
) -> str:
    if resume is not None:
        resume_run_id = normalize_run_id(resume.run_id)
        if requested_run_id is not None:
            requested_run_id = normalize_run_id(requested_run_id)
            if requested_run_id != resume_run_id:
                raise InvalidResumeRequestError(
                    f"metadata.{RUN_METADATA_KEY} does not match the interrupt tool call."
                )
        return resume_run_id

    if requested_run_id is not None:
        return normalize_run_id(requested_run_id)
    return str(uuid.uuid4())


def normalize_run_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise InvalidRunIDError(
            f"metadata.{RUN_METADATA_KEY} must be a UUID when provided."
        ) from exc
    if parsed.int == 0:
        raise InvalidRunIDError(
            f"metadata.{RUN_METADATA_KEY} must not be the nil UUID."
        )
    return str(parsed)


def get_run_id(request: ChatCompletionRequest) -> str | None:
    return (request.metadata or {}).get(RUN_METADATA_KEY)


def normalize_checkpoint_scope(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphConfigurationError(
            "checkpoint_scope must resolve to a non-empty server-trusted string."
        )
    return value.strip()


def checkpoint_key(model: str, run_id: str, *, scope: str = "default") -> str:
    """Derive a fixed-length storage key scoped to this protocol and model."""
    identity = json.dumps(
        ["langgraph-openai-serve.interrupt.v2", scope, model, run_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


async def get_checkpoint_tuple(
    graph: CompiledStateGraph,
    runnable_config: RunnableConfig,
) -> CheckpointTuple | None:
    checkpointer = cast(BaseCheckpointSaver, graph.checkpointer)
    return await checkpointer.aget_tuple(runnable_config)


def checkpoint_state_token(checkpoint_tuple: CheckpointTuple) -> str:
    """Bind a tool call to a checkpoint and its durable resume generation.

    Sequential interrupts in one LangGraph task may reuse both interrupt ID and
    checkpoint ID. LangGraph records preceding answers as resume-channel pending
    writes, so their per-task lengths provide the missing durable generation.
    """
    checkpoint_id = require_checkpoint_id(checkpoint_tuple.config)
    resume_generations = sorted(
        (
            task_id,
            len(value) if isinstance(value, (list, tuple)) else 1,
        )
        for task_id, channel, value in checkpoint_tuple.pending_writes or ()
        if channel == RESUME
    )
    identity = json.dumps(
        [
            "langgraph-openai-serve.interrupt-state.v1",
            checkpoint_id,
            resume_generations,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def require_checkpoint_id(config: RunnableConfig) -> str:
    """Return the checkpoint id from a validated LangGraph config."""
    try:
        checkpoint_id = get_checkpoint_id(config)
    except (AttributeError, KeyError, TypeError):
        checkpoint_id = None
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise RuntimeError("Durable interrupt state has no checkpoint_id.")
    return checkpoint_id


def build_runnable_config(
    callbacks: Callbacks,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig | None:
    if settings.ENABLE_LANGFUSE is True:
        # GraphConfig is shared across requests; add tracing without mutating its
        # callback collection or manager.
        if callbacks is None:
            callbacks = [langfuse_handler]
        elif isinstance(callbacks, list):
            callbacks = [
                *cast(list[BaseCallbackHandler], callbacks),
                langfuse_handler,
            ]
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
