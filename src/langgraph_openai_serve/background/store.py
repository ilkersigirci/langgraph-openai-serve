"""OpenAI-visible persistence for polling-only background Responses."""

from __future__ import annotations

from datetime import (  # ruff: ignore[typing-only-standard-library-import] - Pydantic resolves model annotations at runtime.
    datetime,
    timedelta,
)
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, JsonValue

if TYPE_CHECKING:
    from collections.abc import Sequence


class ResponseStatus(StrEnum):
    """Statuses visible through the OpenAI Response object."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {
        ResponseStatus.COMPLETED,
        ResponseStatus.INCOMPLETE,
        ResponseStatus.FAILED,
        ResponseStatus.CANCELLED,
    }
)


class BackgroundStoreError(RuntimeError):
    """Base class for expected persistence-boundary decisions."""


class BackgroundIdempotencyConflictError(BackgroundStoreError):
    """A create idempotency key was reused with different request content."""


class BackgroundResponseExpiredError(BackgroundStoreError):
    """The idempotency reservation remains but its Response has expired."""


class BackgroundCapacityError(BackgroundStoreError):
    """The configured active-response admission limit was reached."""


class NewRun(BaseModel):
    """Complete data persisted before a native workflow is submitted."""

    response_id: str
    owner_scope: str
    model: str
    checkpoint_thread_id: str
    graph_version: str
    envelope: dict[str, JsonValue]
    request_fingerprint: str
    idempotency_digest: str | None
    response: dict[str, JsonValue]
    created_at: datetime
    initial_call_ids: tuple[str, ...] = ()
    initial_message_count: int = 0

    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredRun(BaseModel):
    """Immutable snapshot of one authoritative Response record."""

    response_id: str
    owner_scope: str
    model: str
    checkpoint_thread_id: str
    graph_version: str
    envelope: dict[str, JsonValue]
    request_fingerprint: str
    idempotency_digest: str | None
    response: dict[str, JsonValue] | None
    status: ResponseStatus
    created_at: datetime
    updated_at: datetime
    workflow_run_id: str | None = None
    terminal_at: datetime | None = None
    result_expires_at: datetime | None = None
    idempotency_expires_at: datetime | None = None
    initial_call_ids: tuple[str, ...] = ()
    initial_message_count: int = 0
    cancellation_pending: bool = False
    cleanup_pending: bool = False

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def terminal(self) -> bool:
        """Whether the public Response can no longer change."""
        return self.status in TERMINAL_STATUSES


@runtime_checkable
class ResponseStore(Protocol):
    """Atomic Response persistence used by a background backend and worker."""

    async def accept(self, run: NewRun, *, capacity: int) -> StoredRun:
        """Commit a queued Response or resolve its idempotency key."""
        ...

    async def record_workflow_run(
        self,
        response_id: str,
        workflow_run_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Store the native workflow ID used for cancellation."""
        ...

    async def claim_pending_submissions(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Claim accepted or cancelled runs without a native workflow receipt."""
        ...

    async def get(
        self,
        response_id: str,
        owner_scope: str,
        *,
        now: datetime | None = None,
    ) -> StoredRun | None:
        """Read one authorized, unexpired public Response snapshot."""
        ...

    async def get_internal(self, response_id: str) -> StoredRun | None:
        """Read one run for trusted backend or worker code."""
        ...

    async def mark_in_progress(
        self,
        response_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Move an active queued Response to in-progress."""
        ...

    async def request_cancellation(  # ruff: ignore[too-many-arguments] - One atomic transition carries authorization and retention.
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
        idempotency_retention: timedelta,
    ) -> StoredRun | None:
        """Atomically choose cancellation unless a terminal result won."""
        ...

    async def publish_terminal(
        self,
        response_id: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
        idempotency_retention: timedelta,
    ) -> StoredRun | None:
        """Commit the immutable terminal Response if the run is active."""
        ...

    async def claim_cancellations(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Claim cancelled runs whose native cancellation needs delivery."""
        ...

    async def finish_cancellation(self, response_id: str, *, now: datetime) -> bool:
        """Record successful delivery of a native cancellation."""
        ...

    async def claim_cleanup_ready(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Claim terminal checkpoint lineages for one cleanup attempt."""
        ...

    async def finish_cleanup(self, response_id: str, *, now: datetime) -> bool:
        """Record successful removal of one checkpoint lineage."""
        ...

    async def abandon_cleanup(self, response_id: str, *, now: datetime) -> bool:
        """Stop retrying cleanup when its graph configuration is unavailable."""
        ...

    async def expire(self, *, now: datetime, limit: int) -> int:
        """Expire terminal results and idempotency tombstones."""
        ...


def terminal_run(
    run: StoredRun,
    response: dict[str, JsonValue],
    *,
    now: datetime,
    result_retention: timedelta,
    idempotency_retention: timedelta,
) -> StoredRun:
    """Build the shared terminal transition used by store adapters."""
    status = ResponseStatus(str(response["status"]))
    if status not in TERMINAL_STATUSES:
        msg = "A published background Response must be terminal."
        raise ValueError(msg)
    result_expires_at = now + result_retention
    idempotency_expires_at = (
        max(result_expires_at, now + idempotency_retention)
        if run.idempotency_digest is not None
        else None
    )
    return run.model_copy(
        update={
            "response": response,
            "status": status,
            "terminal_at": now,
            "result_expires_at": result_expires_at,
            "idempotency_expires_at": idempotency_expires_at,
            "cancellation_pending": status is ResponseStatus.CANCELLED,
            "cleanup_pending": True,
            "updated_at": now,
        }
    )


def tombstone_run(run: StoredRun, *, now: datetime) -> StoredRun:
    """Remove request/result payloads while retaining cleanup and key metadata."""
    return run.model_copy(
        update={
            "response": None,
            "envelope": {},
            "initial_call_ids": (),
            "initial_message_count": 0,
            "updated_at": now,
        }
    )


def expired_run_deletable(run: StoredRun, *, now: datetime) -> bool:
    """Return whether neither idempotency nor checkpoint cleanup needs the row."""
    idempotency_retained = bool(
        run.idempotency_digest is not None
        and run.idempotency_expires_at is not None
        and run.idempotency_expires_at > now
    )
    return (
        not idempotency_retained
        and not run.cancellation_pending
        and not run.cleanup_pending
    )


__all__ = [
    "BackgroundCapacityError",
    "BackgroundIdempotencyConflictError",
    "BackgroundResponseExpiredError",
    "NewRun",
    "ResponseStatus",
    "ResponseStore",
    "StoredRun",
]
