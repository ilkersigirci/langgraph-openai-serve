"""Durable state and resume handling for interrupt-enabled graph runs."""

import hashlib
import json
import uuid
from typing import Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    RESUME,
    BaseCheckpointSaver,
    get_checkpoint_id,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StateSnapshot

from langgraph_openai_serve.graph.graph_registry import (
    GraphConfig,
    GraphConfigurationError,
)
from langgraph_openai_serve.graph.interrupt.errors import InvalidResumeRequestError
from langgraph_openai_serve.graph.interrupt.models import (
    InterruptResume,
    LangGraphInterruptBatch,
)
from langgraph_openai_serve.graph.interrupt.validation import validate_interrupt_payload
from langgraph_openai_serve.graph.request import GraphRequest

RUN_METADATA_KEY = "langgraph_run_id"


class InvalidRunIDError(ValueError):
    """Raised when a caller-supplied run id is not a UUID."""


class InterruptStateConflictError(RuntimeError):
    """Raised when a resume does not match durable pending state."""


async def prepare_interrupt_input(  # ruff: ignore[too-many-arguments]
    graph_config: GraphConfig,
    graph: CompiledStateGraph,
    request: GraphRequest,
    snapshot: StateSnapshot,
    resume: InterruptResume | None,
    *,
    messages: list[BaseMessage],
) -> tuple[Any, bool]:
    """Build a new input or causally validate an interrupt resume."""
    pending_interrupts = interrupts_by_id(snapshot)
    checkpoint_id = get_checkpoint_id(snapshot.config)

    if resume is None:
        if checkpoint_id is None:
            return await graph_config.build_input(request, messages), True
        if pending_interrupts:
            # Re-emit persisted tool calls without rerunning graph nodes.
            return None, False
        msg = "This run_id has already been used."
        raise InterruptStateConflictError(msg)

    if checkpoint_id is None:
        msg = "No durable interrupt state exists for this run."
        raise InterruptStateConflictError(msg)
    if not pending_interrupts:
        msg = "This run no longer has pending interrupts."
        raise InterruptStateConflictError(msg)

    state_token = await checkpoint_state_token(graph, snapshot.config)
    if state_token is None:
        msg = "No durable interrupt state exists for this run."
        raise InterruptStateConflictError(msg)
    return _resume_interrupt_inputs(
        state_token,
        set(pending_interrupts),
        resume,
    ), True


def _resume_interrupt_inputs(
    state_token: str,
    pending_ids: set[str],
    resume: InterruptResume,
) -> Command:
    if resume.state_token != state_token:
        msg = "The interrupt result is stale for the current interrupt generation."
        raise InterruptStateConflictError(msg)
    if set(resume.values) != pending_ids:
        msg = "Interrupt results do not match the complete pending interrupt set."
        raise InterruptStateConflictError(msg)

    # Always use the ID/value form, including for one interrupt. It preserves
    # OpenAI tool_call causality and handles JSON null as a legitimate answer.
    return Command(resume=resume.values)


def interrupts_by_id(snapshot: StateSnapshot) -> dict[str, Interrupt]:
    """Validate and index the interrupts exposed by a state snapshot."""
    pending: dict[str, Interrupt] = {}
    for interrupt in snapshot.interrupts:
        interrupt_id = interrupt.id
        if not isinstance(interrupt_id, str) or not interrupt_id:
            msg = "Durable interrupt state has an invalid interrupt id."
            raise RuntimeError(msg)
        if interrupt_id in pending:
            msg = "Durable interrupt state has duplicate interrupt ids."
            raise RuntimeError(msg)
        pending[interrupt_id] = interrupt
    return pending


def resolve_run_id(
    requested_run_id: str | None,
    resume: InterruptResume | None,
) -> str:
    """Resolve and validate the durable run identity for a request."""
    if resume is not None:
        resume_run_id = normalize_run_id(resume.run_id)
        if requested_run_id is not None:
            requested_run_id = normalize_run_id(requested_run_id)
            if requested_run_id != resume_run_id:
                msg = (
                    f"metadata.{RUN_METADATA_KEY} does not match the interrupt "
                    "tool call."
                )
                raise InvalidResumeRequestError(msg)
        return resume_run_id

    if requested_run_id is not None:
        return normalize_run_id(requested_run_id)
    return str(uuid.uuid4())


def normalize_run_id(value: str) -> str:
    """Return the canonical form of a valid, non-nil UUID run id."""
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        msg = f"metadata.{RUN_METADATA_KEY} must be a UUID when provided."
        raise InvalidRunIDError(msg) from exc
    if parsed.int == 0:
        msg = f"metadata.{RUN_METADATA_KEY} must not be the nil UUID."
        raise InvalidRunIDError(msg)
    return str(parsed)


def get_run_id(request: GraphRequest) -> str | None:
    """Read the optional interrupt run id from normalized request metadata."""
    return request.metadata.get(RUN_METADATA_KEY)


def normalize_checkpoint_scope(value: str) -> str:
    """Validate a server-owned checkpoint isolation scope."""
    if not isinstance(value, str) or not value.strip():
        msg = "checkpoint_scope must resolve to a non-empty server-trusted string."
        raise GraphConfigurationError(msg)
    return value.strip()


def checkpoint_key(model: str, run_id: str, *, scope: str = "default") -> str:
    """Derive a fixed-length storage key scoped to this protocol and model."""
    identity = json.dumps(
        ["langgraph-openai-serve.interrupt.v2", scope, model, run_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


async def checkpoint_state_token(
    graph: CompiledStateGraph,
    runnable_config: RunnableConfig,
) -> str | None:
    """
    Fingerprint the latest checkpoint in every namespace.

    Nested resumes may not advance the root checkpoint, and indirectly invoked
    subgraphs are not exposed through state snapshots. Scanning the checkpointer
    keeps stale-resume detection generic without introducing separate state.

    Performance impact: Local PostgreSQL measurements were 0.5-0.7 ms for the
    current 1-2 tuple runs, scaling linearly to about 5 ms at 100 and 45 ms at
    1,000 small tuples.
    """
    checkpointer = cast("BaseCheckpointSaver", graph.checkpointer)
    thread_id = runnable_config["configurable"]["thread_id"]
    heads: dict[str, tuple[str, list[tuple[str, int]]]] = {}

    async for checkpoint_tuple in checkpointer.alist(
        {"configurable": {"thread_id": thread_id}}
    ):
        namespace = checkpoint_tuple.config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = require_checkpoint_id(checkpoint_tuple.config)
        head = heads.get(namespace)
        if head is not None and checkpoint_id <= head[0]:
            continue

        heads[namespace] = (
            checkpoint_id,
            sorted(
                (
                    task_id,
                    len(value) if isinstance(value, (list, tuple)) else 1,
                )
                for task_id, channel, value in checkpoint_tuple.pending_writes or ()
                if channel == RESUME
            ),
        )

    if not heads:
        return None

    identity = json.dumps(
        [
            "langgraph-openai-serve.interrupt-state.v2",
            sorted((namespace, *head) for namespace, head in heads.items()),
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
        msg = "Durable interrupt state has no checkpoint_id."
        raise RuntimeError(msg)
    return checkpoint_id


async def durable_interrupt_batch(
    graph: CompiledStateGraph,
    runnable_config: RunnableConfig | None,
    run_id: str | None,
) -> LangGraphInterruptBatch | None:
    """Read the durable checkpoint head after graph execution has quiesced."""
    if runnable_config is None:
        msg = "Interrupt-enabled runs require runnable configuration."
        raise RuntimeError(msg)

    snapshot = await graph.aget_state(runnable_config, subgraphs=True)
    if not snapshot.interrupts:
        return None

    pending_interrupts = interrupts_by_id(snapshot)
    for interrupt in pending_interrupts.values():
        validate_interrupt_payload(interrupt.value)

    if run_id is None:
        msg = "run_id cannot be None"
        raise RuntimeError(msg)
    state_token = await checkpoint_state_token(graph, snapshot.config)
    if state_token is None:
        msg = "Interrupted LangGraph state has no checkpoint tuple."
        raise RuntimeError(msg)
    return LangGraphInterruptBatch(
        run_id=run_id,
        state_token=state_token,
        interrupts=tuple(pending_interrupts.values()),
    )
