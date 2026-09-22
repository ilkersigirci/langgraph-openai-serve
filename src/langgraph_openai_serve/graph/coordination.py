"""Nonblocking coordination for checkpointed graph runs."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, runtime_checkable


class RunBusyError(RuntimeError):
    """Raised when a checkpoint thread or local lease capacity is occupied."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__("This graph run cannot acquire its coordination lease.")


class RunLeaseLostError(RuntimeError):
    """Raised when execution no longer owns its checkpoint-thread lease."""


@dataclass(slots=True)
class RunLease:
    """
    Report irrevocable loss of ownership for one acquired lease.

    Coordinators set ``lost`` before cancelling the owning task. LGOS also
    checks it before publication and cleanup, including when application code
    suppresses cancellation. This cooperative check is not storage fencing.
    A new acquisition must yield a new lease; never reset a lost lease.
    """

    lost: bool = False

    def ensure_owned(self) -> None:
        """Reject further work after the coordinator reports ownership loss."""
        if self.lost:
            msg = "The graph run lost its coordination lease."
            raise RunLeaseLostError(msg)


@runtime_checkable
class RunCoordinator(Protocol):
    """
    Coordinate execution and cleanup of one checkpoint thread.

    All participants sharing checkpoint state must share the coordination
    namespace. Reject an occupied key with ``RunBusyError`` instead of waiting
    for its owner. Different keys may run concurrently, subject to capacity.
    Yield a fresh ``RunLease`` and release only that acquisition on every exit,
    including cancellation. If ownership becomes uncertain, mark the lease
    lost before cancelling its owning task and propagate failure on exit.

    Implementations own their failure and deployment assumptions; satisfying
    this protocol alone does not guarantee distributed mutual exclusion.
    """

    def __call__(
        self,
        key: str,
        /,
    ) -> AbstractAsyncContextManager[RunLease]:
        """Return a context manager that acquires the lease on async entry."""
        ...


class InMemoryRunCoordinator:
    """Coordinate graph runs within one process without waiting."""

    def __init__(self) -> None:
        self._active_keys: set[str] = set()
        self._guard = Lock()

    @asynccontextmanager
    async def __call__(self, key: str, /) -> AsyncIterator[RunLease]:
        """
        Acquire lease asynchronously.

        Yields:
            The process-local state of the acquired lease.

        """
        self._acquire(key)
        try:
            yield RunLease()
        finally:
            self._release(key)

    def _acquire(self, key: str) -> None:
        with self._guard:
            if key in self._active_keys:
                raise RunBusyError(key)
            self._active_keys.add(key)

    def _release(self, key: str) -> None:
        with self._guard:
            self._active_keys.remove(key)


__all__ = [
    "InMemoryRunCoordinator",
    "RunBusyError",
    "RunCoordinator",
    "RunLease",
    "RunLeaseLostError",
]
