"""Nonblocking coordination for checkpointed graph runs."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, runtime_checkable


class RunBusyError(RuntimeError):
    """Raised when an interrupt run cannot acquire its coordination lease."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__("This graph run cannot acquire its coordination lease.")


@dataclass(slots=True)
class RunLease:
    """Expose whether the exact coordination session was lost while owned."""

    lost: bool = False


@runtime_checkable
class RunCoordinator(Protocol):
    """Acquire a lease that rejects rather than queues an occupied interrupt run."""

    def __call__(
        self,
        key: str,
        /,
    ) -> AbstractAsyncContextManager[RunLease | None]:
        """Acquire lease synchronously."""
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


__all__ = ["InMemoryRunCoordinator", "RunBusyError", "RunCoordinator", "RunLease"]
