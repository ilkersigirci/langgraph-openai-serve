"""Durable state and resume handling for interrupt-enabled graph runs."""

import hashlib
import json
import uuid
from collections.abc import Iterable

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from langgraph_openai_serve.graph.graph_registry import GraphConfigurationError
from langgraph_openai_serve.graph.interrupt.errors import InvalidResumeRequestError
from langgraph_openai_serve.graph.interrupt.models import (
    InterruptResume,
    LangGraphInterruptBatch,
)
from langgraph_openai_serve.graph.interrupt.validation import validate_interrupt_payload
from langgraph_openai_serve.graph.request import GraphRequest
from langgraph_openai_serve.protocol import RUN_METADATA_KEY


class InvalidRunIDError(ValueError):
    """Raised when a caller-supplied run id is not a UUID."""


class InterruptStateConflictError(RuntimeError):
    """Raised when a resume does not match durable pending state."""


async def prepare_interrupt_state(
    graph: CompiledStateGraph,
    runnable_config: RunnableConfig,
    run_id: str,
    resume: InterruptResume | None,
) -> Command | LangGraphInterruptBatch | None:
    """Return a validated resume, a pending retry batch, or a fresh-run marker."""
    snapshot = await graph.aget_state(runnable_config, subgraphs=True)
    if get_checkpoint_id(snapshot.config) is None:
        if resume is None:
            return None
        msg = "No durable interrupt state exists for this run."
        raise InterruptStateConflictError(msg)

    batch = interrupt_batch(snapshot.interrupts, run_id)
    if batch is None:
        msg = "This run no longer has pending interrupts."
        raise InterruptStateConflictError(msg)
    if resume is None:
        return batch
    if set(resume.values) != {item.id for item in batch.interrupts}:
        msg = "Interrupt results do not match the complete pending interrupt set."
        raise InterruptStateConflictError(msg)

    # Native ID/value resumes answer parallel interrupts without replaying input.
    return Command(resume=resume.values)


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
                    "Response."
                )
                raise InvalidResumeRequestError(
                    msg,
                    param=f"metadata.{RUN_METADATA_KEY}",
                )
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


def interrupt_batch(
    interrupts: Iterable[Interrupt],
    run_id: str | None,
) -> LangGraphInterruptBatch | None:
    """Validate one native pending interrupt set."""
    pending_interrupts = _native_interrupts_by_id(interrupts)
    if not pending_interrupts:
        return None

    if run_id is None:
        msg = "run_id cannot be None"
        raise RuntimeError(msg)
    return LangGraphInterruptBatch(
        run_id=run_id,
        interrupts=tuple(pending_interrupts.values()),
    )


def _native_interrupts_by_id(
    interrupts: Iterable[Interrupt],
) -> dict[str, Interrupt]:
    """Validate and index native interrupt results."""
    pending: dict[str, Interrupt] = {}
    for interrupt in interrupts:
        interrupt_id = interrupt.id
        if not isinstance(interrupt_id, str) or not interrupt_id:
            msg = "Native interrupt result has an invalid interrupt id."
            raise RuntimeError(msg)
        validate_interrupt_payload(interrupt.value)
        if interrupt_id not in pending:
            pending[interrupt_id] = interrupt
        elif pending[interrupt_id] != interrupt:
            msg = "Native interrupt result has conflicting data for one interrupt id."
            raise RuntimeError(msg)
        else:
            msg = "Durable interrupt state has duplicate interrupt ids."
            raise RuntimeError(msg)
    return pending
