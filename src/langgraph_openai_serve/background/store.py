"""OpenAI-visible persistence for polling-only background Responses."""

from __future__ import annotations

from datetime import (  # ruff: ignore[typing-only-standard-library-import] - Pydantic resolves model annotations at runtime.
    datetime,
    timedelta,
)
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

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


class NewRun(BaseModel):
    """Complete data persisted before a background job is submitted."""

    response_id: str
    owner_scope: str
    model: str
    checkpoint_thread_id: str
    envelope: dict[str, JsonValue]
    response: dict[str, JsonValue]
    created_at: datetime
    # Transcript item IDs that precede this Response: the input's, or those of
    # the paused transcript an answer continues.
    prior_ids: tuple[str, ...] = ()
    # Scoped digest of the client's Idempotency-Key and of the request it
    # protects. A replay with the same digest returns the stored run.
    idempotency_digest: str | None = None
    request_fingerprint: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class StoredRun(NewRun):
    """Immutable snapshot of one authoritative Response record."""

    status: ResponseStatus
    updated_at: datetime
    result_expires_at: datetime | None = None
    cleanup_pending: bool = False

    @property
    def terminal(self) -> bool:
        """Whether the public Response can no longer change."""
        return self.status in TERMINAL_STATUSES

    def visible_to(self, owner_scope: str, *, now: datetime) -> bool:
        """Whether an owner may still read this Response."""
        return self.owner_scope == owner_scope and (
            self.result_expires_at is None or self.result_expires_at > now
        )


class ResponseStore(Protocol):
    """
    Atomic Response persistence shared by a backend and its workers.

    Implementations own atomicity across their deployment scope. ``finish``
    must be first-terminal-wins under concurrent calls. The application owns
    initialization, connections, migrations, and shutdown.
    """

    async def create(self, run: NewRun) -> StoredRun:
        """
        Persist a new queued run, or return the run holding its idempotency digest.

        The digest is unique among stored runs, so concurrent creates with one
        digest yield one run. The key lives exactly as long as that run.
        """
        ...

    async def get(self, response_id: str) -> StoredRun | None:
        """Read one run, including expired runs not yet removed."""
        ...

    async def find(self, idempotency_digest: str) -> StoredRun | None:
        """Read the run holding one idempotency digest."""
        ...

    async def mark_in_progress(
        self,
        response_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Move a queued run in progress; return None when it is missing or terminal."""
        ...

    async def finish(
        self,
        response_id: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
    ) -> StoredRun | None:
        """
        Commit a terminal Response unless another terminal outcome won.

        Return the winning run, which is the existing one when it was already
        terminal, or None when the run is missing. Mark checkpoint cleanup
        pending in the same transition.
        """
        ...

    async def claim_queued(
        self,
        *,
        created_before: datetime,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """
        Return runs still queued since before ``created_before``.

        Touch ``updated_at`` like ``claim_cleanup_ready`` so a backlog of
        legitimately queued runs cannot starve later ones.
        """
        ...

    async def claim_cleanup_ready(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """
        Return terminal runs whose checkpoints still need deletion.

        Touch ``updated_at`` on returned runs so a failing run cannot starve
        later ones. Claims are not exclusive leases.
        """
        ...

    async def finish_cleanup(self, response_id: str, *, now: datetime) -> None:
        """Record that checkpoint cleanup is done or cannot be performed."""
        ...

    async def expire(self, *, now: datetime, limit: int) -> int:
        """Delete at most ``limit`` expired runs without pending cleanup."""
        ...


def terminal_status(response: dict[str, JsonValue]) -> ResponseStatus:
    """Return the status of a Response that a store may finish with."""
    status = ResponseStatus(str(response["status"]))
    if status not in TERMINAL_STATUSES:
        msg = "A finished background Response must be terminal."
        raise ValueError(msg)
    return status


__all__ = [
    "NewRun",
    "ResponseStatus",
    "ResponseStore",
    "StoredRun",
]
