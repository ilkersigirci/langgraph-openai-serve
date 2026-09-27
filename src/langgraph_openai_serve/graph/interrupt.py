"""Durable LangGraph interrupts: run leases, run identity, and resume state."""

import hashlib
import json
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from langgraph_openai_serve.core.errors import (
    GraphError,
    InvalidRequestError,
)
from langgraph_openai_serve.protocol import RUN_METADATA_KEY


class RunBusyError(InvalidRequestError):
    """Raised when another request holds an interrupt run's lease."""

    def __init__(self) -> None:
        super().__init__(
            "This interrupt run cannot acquire its coordination lease.",
            code="run_busy",
            status_code=409,
        )


@runtime_checkable
class RunCoordinator(Protocol):
    """Acquire a lease that rejects, rather than queues, an occupied run."""

    def __call__(self, key: str, /) -> AbstractAsyncContextManager[None]:
        """Hold the lease for ``key`` or raise ``RunBusyError``."""
        ...


class InMemoryRunCoordinator:
    """Coordinate interrupt runs within one event loop."""

    def __init__(self) -> None:
        self._active: set[str] = set()

    @asynccontextmanager
    async def __call__(self, key: str, /) -> AsyncIterator[None]:
        """Hold the lease for ``key`` or raise ``RunBusyError``."""
        if key in self._active:
            raise RunBusyError
        self._active.add(key)
        try:
            yield
        finally:
            self._active.discard(key)


@dataclass(frozen=True, slots=True)
class InterruptResume:
    """A complete set of interrupt answers for one run."""

    run_id: str
    values: dict[str, str]


@dataclass(frozen=True)
class LangGraphInterruptBatch:
    """The durable interrupts awaiting answers for one graph run."""

    run_id: str
    interrupts: tuple[Interrupt, ...]


async def prepare_interrupt_state(
    graph: CompiledStateGraph,
    config: RunnableConfig,
    run_id: str,
    resume: InterruptResume | None,
) -> Command | LangGraphInterruptBatch | None:
    """Return a validated resume, a pending retry batch, or None for a new run."""
    snapshot = await graph.aget_state(config, subgraphs=True)
    if get_checkpoint_id(snapshot.config) is None:
        if resume is None:
            return None
        msg = "No durable interrupt state exists for this run."
        raise _state_conflict(msg)

    if not snapshot.interrupts:
        msg = "This run no longer has pending interrupts."
        raise _state_conflict(msg)
    batch = interrupt_batch(snapshot.interrupts, run_id)
    if resume is None:
        return batch
    if set(resume.values) != {item.id for item in batch.interrupts}:
        msg = "Interrupt results do not match the complete pending interrupt set."
        raise _state_conflict(msg)

    # Native ID/value resumes answer parallel interrupts without replaying input.
    return Command(resume=resume.values)


def resolve_run_id(requested_run_id: str | None, resume: InterruptResume | None) -> str:
    """Resolve and validate the durable run identity for a request."""
    if resume is None:
        return (
            str(uuid.uuid4()) if requested_run_id is None else _run_id(requested_run_id)
        )
    if requested_run_id is not None and _run_id(requested_run_id) != resume.run_id:
        msg = f"metadata.{RUN_METADATA_KEY} does not match the interrupt Response."
        raise InvalidRequestError(msg, param=f"metadata.{RUN_METADATA_KEY}")
    return resume.run_id


def checkpoint_key(model: str, run_id: str, *, scope: str = "default") -> str:
    """Derive a fixed-length checkpoint thread ID for one scoped model run."""
    identity = json.dumps(
        ["langgraph-openai-serve.interrupt", scope, model, run_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def interrupt_batch(
    interrupts: Iterable[Interrupt],
    run_id: str,
) -> LangGraphInterruptBatch:
    """Validate pending interrupts as JSON function-call arguments."""
    pending = tuple(interrupts)
    for item in pending:
        if not isinstance(item.value, dict):
            msg = "LangGraph interrupt payloads must be JSON objects."
            raise GraphError(msg)
        try:
            json.dumps(item.value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            msg = "LangGraph interrupt payloads must be valid JSON values."
            raise GraphError(msg) from exc
    return LangGraphInterruptBatch(run_id=run_id, interrupts=pending)


def _run_id(value: str) -> str:
    """Return the canonical form of a caller-selected, non-nil UUID."""
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        msg = f"metadata.{RUN_METADATA_KEY} must be a UUID when provided."
        raise InvalidRequestError(msg, param=f"metadata.{RUN_METADATA_KEY}") from exc
    if parsed.int == 0:
        msg = f"metadata.{RUN_METADATA_KEY} must not be the nil UUID."
        raise InvalidRequestError(msg, param=f"metadata.{RUN_METADATA_KEY}")
    return str(parsed)


def _state_conflict(message: str) -> InvalidRequestError:
    return InvalidRequestError(
        message,
        param="input",
        code="interrupt_state_conflict",
        status_code=409,
    )


__all__ = [
    "InMemoryRunCoordinator",
    "InterruptResume",
    "LangGraphInterruptBatch",
    "RunBusyError",
    "RunCoordinator",
]
