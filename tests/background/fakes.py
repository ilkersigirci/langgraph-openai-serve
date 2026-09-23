"""Process-local background fakes shared by package tests."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from langgraph_openai_serve.background.contracts import BackgroundSettings
from langgraph_openai_serve.background.in_memory import InMemoryResponseStore

if TYPE_CHECKING:
    from langgraph_openai_serve.background.store import StoredRun


class MemoryBackgroundBackend:
    """Manually driven backend: tests receive submitted jobs and run them."""

    def __init__(
        self,
        *,
        store: InMemoryResponseStore | None = None,
        settings: BackgroundSettings | None = None,
    ) -> None:
        self.store = store or InMemoryResponseStore()
        self.settings = settings or BackgroundSettings()
        self.submit_error: Exception | None = None
        self.stopped: list[str] = []
        self._pending: deque[str] = deque()

    async def submit(self, run: StoredRun) -> None:
        if self.submit_error is not None:
            raise self.submit_error
        self._pending.append(run.response_id)

    async def stop(self, run: StoredRun) -> None:
        self.stopped.append(run.response_id)

    async def receive(self) -> str | None:
        while self._pending:
            job = self._pending.popleft()
            if job not in self.stopped:
                return job
        return None


__all__ = ["MemoryBackgroundBackend"]
