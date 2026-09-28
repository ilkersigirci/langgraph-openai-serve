"""Durable LangGraph interrupts: run leases, run identity, and resume state."""

import hashlib
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from langgraph_openai_serve.core.errors import (
    GraphError,
    InvalidRequestError,
)

# LangGraph copies run metadata into each checkpoint, which marks LGOS runs.
OPERATION_ID_METADATA_KEY = "lgos.operation_id"


class RunBusyError(InvalidRequestError):
    """Raised when another request holds an interrupt run's lease."""

    def __init__(self) -> None:
        super().__init__(
            "This interrupt run cannot acquire its coordination lease.",
            code="run_busy",
            status_code=409,
        )


class RunCoordinator(Protocol):
    """
    Acquire a lease that rejects, rather than queues, an occupied run.

    Exiting the lease must release it even when the exit is cancelled, because
    run cleanup is abandoned after a deadline; shield an asynchronous release.
    """

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


async def resume_command(
    graph: CompiledStateGraph,
    config: RunnableConfig,
    resume: InterruptResume,
) -> Command:
    """Validate answers against the run's durable pending interrupts."""
    snapshot = await graph.aget_state(config, subgraphs=True)
    if not snapshot.interrupts:
        msg = "This run has no pending interrupts."
        raise _state_conflict(msg)
    if set(resume.values) != {item.id for item in snapshot.interrupts}:
        msg = "Interrupt results do not match the complete pending interrupt set."
        raise _state_conflict(msg)
    # Native ID/value resumes answer parallel interrupts without replaying input.
    return Command(resume=resume.values)


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


async def delete_expired_interrupt_runs(
    checkpointer: BaseCheckpointSaver,
    run_coordinator: RunCoordinator,
    *,
    older_than: timedelta,
) -> int:
    """
    Delete paused interrupt runs whose latest pause is older than ``older_than``.

    Other threads in ``checkpointer`` are left alone, and a run whose lease is
    held, such as one being resumed, is skipped until the next call.

    Returns:
        The number of deleted runs.

    """
    cutoff = datetime.now(UTC) - older_than
    paused_at: dict[str, datetime] = {}
    async for item in checkpointer.alist(None):
        if OPERATION_ID_METADATA_KEY in item.metadata:
            thread_id = item.config["configurable"]["thread_id"]
            timestamp = datetime.fromisoformat(item.checkpoint["ts"])
            paused_at[thread_id] = max(timestamp, paused_at.get(thread_id, timestamp))

    deleted = 0
    for thread_id, timestamp in paused_at.items():
        if timestamp >= cutoff:
            continue
        try:
            async with run_coordinator(thread_id):
                # The run may have been resumed, or deleted, since it was listed.
                latest = await checkpointer.aget_tuple(
                    {"configurable": {"thread_id": thread_id}}
                )
                if latest is None or (
                    datetime.fromisoformat(latest.checkpoint["ts"]) >= cutoff
                ):
                    continue
                await checkpointer.adelete_thread(thread_id)
        except RunBusyError:
            continue
        deleted += 1
    return deleted


def _state_conflict(message: str) -> InvalidRequestError:
    return InvalidRequestError(
        message,
        param="input",
        code="interrupt_state_conflict",
        status_code=409,
    )


__all__ = [
    "OPERATION_ID_METADATA_KEY",
    "InMemoryRunCoordinator",
    "InterruptResume",
    "LangGraphInterruptBatch",
    "RunBusyError",
    "RunCoordinator",
    "delete_expired_interrupt_runs",
]
