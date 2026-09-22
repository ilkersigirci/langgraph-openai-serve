"""Process-local background fakes shared by package tests."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anyio import Lock

from langgraph_openai_serve.background.contracts import BackgroundSettings
from langgraph_openai_serve.background.in_memory import InMemoryResponseStore
from langgraph_openai_serve.background.store import (
    NewRun,
    ResponseStatus,
    StoredRun,
)

if TYPE_CHECKING:
    from pydantic import JsonValue


class MemoryBackgroundBackend:
    """Manually driven lifecycle backend for API tests."""

    def __init__(
        self,
        *,
        store: InMemoryResponseStore | None = None,
        settings: BackgroundSettings | None = None,
    ) -> None:
        self.store = store or InMemoryResponseStore()
        self.settings = settings or BackgroundSettings()
        self._pending: deque[str] = deque()
        self._cancelled: set[str] = set()
        self._lock = Lock()

    async def create(self, run: NewRun) -> StoredRun:
        accepted = await self.store.accept(
            run,
            capacity=self.settings.admission_capacity,
        )
        if accepted.workflow_run_id is not None:
            return accepted
        recorded = await self.store.record_workflow_run(
            accepted.response_id,
            accepted.response_id,
            now=datetime.now(UTC),
        )
        if recorded is None:
            msg = "Test workflow receipt could not be persisted."
            raise RuntimeError(msg)
        async with self._lock:
            self._pending.append(recorded.response_id)
        return recorded

    async def retrieve(self, response_id: str, owner_scope: str) -> StoredRun | None:
        return await self.store.get(response_id, owner_scope)

    async def cancel(
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        stored: bool,
    ) -> StoredRun | None:
        result = await self.store.request_cancellation(
            response_id,
            owner_scope,
            response,
            now=datetime.now(UTC),
            result_retention=self.settings.result_retention_for(stored=stored),
            idempotency_retention=self.settings.idempotency_retention,
        )
        if result is not None and result.status is ResponseStatus.CANCELLED:
            async with self._lock:
                self._cancelled.add(result.response_id)
        return result

    async def receive(self) -> str | None:
        async with self._lock:
            while self._pending:
                job = self._pending.popleft()
                if job not in self._cancelled:
                    return job
            return None


__all__ = ["MemoryBackgroundBackend"]
