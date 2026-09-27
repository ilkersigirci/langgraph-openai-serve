"""Durable LangGraph interrupts: run leases, run identity, and resume state."""

import hashlib
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from langgraph_openai_serve.core.errors import (
    GraphError,
    InvalidRequestError,
)


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
