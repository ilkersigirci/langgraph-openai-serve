"""Nonblocking coordination for graph runs that share durable state."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from threading import Lock
from typing import Protocol, runtime_checkable


class RunBusyError(RuntimeError):
    """Raised when a run cannot acquire its coordination lease."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__("This interrupt run cannot acquire its coordination lease.")


@runtime_checkable
class RunCoordinator(Protocol):
    """Acquire a lease that rejects rather than queues an occupied run key."""

    def __call__(
        self,
        key: str,
        /,
    ) -> AbstractAsyncContextManager[None]:
        """Acquire lease synchronously."""
        ...


class InMemoryRunCoordinator:
    """Coordinate runs within one process without waiting on occupied keys."""

    def __init__(self) -> None:
        self._active_keys: set[str] = set()
        self._guard = Lock()

    @asynccontextmanager
    async def __call__(self, key: str, /) -> AsyncIterator[None]:
        """Acquire lease asynchronously."""
        self._acquire(key)
        try:
            yield
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


__all__ = ["InMemoryRunCoordinator", "RunBusyError", "RunCoordinator"]
