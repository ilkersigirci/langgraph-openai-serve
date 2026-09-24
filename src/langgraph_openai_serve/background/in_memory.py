"""Single-process background execution for development and examples."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from anyio import CancelScope, Lock, create_task_group, sleep

from langgraph_openai_serve.background.contracts import BackgroundSettings
from langgraph_openai_serve.background.store import (
    NewRun,
    ResponseStatus,
    StoredRun,
    terminal_status,
)
from langgraph_openai_serve.background.worker import BackgroundWorker
from langgraph_openai_serve.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from anyio.abc import TaskGroup
    from pydantic import JsonValue

    from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = get_logger(__name__)


class InMemoryResponseStore:
    """Keep Response state in one process until it exits."""

    def __init__(self) -> None:
        self._runs: dict[str, StoredRun] = {}
        self._idempotency: dict[str, str] = {}
        self._lock = Lock()

    async def create(self, run: NewRun) -> StoredRun:
        """Persist a new queued run, or return the run holding its digest."""
        async with self._lock:
            digest = run.idempotency_digest
            if digest is not None and digest in self._idempotency:
                return self._copy(self._runs[self._idempotency[digest]])
            if digest is not None:
                self._idempotency[digest] = run.response_id
            return self._put(
                StoredRun(
                    **run.model_dump(),
                    status=ResponseStatus.QUEUED,
                    updated_at=run.created_at,
                )
            )

    async def get(self, response_id: str) -> StoredRun | None:
        """Read one run."""
        async with self._lock:
            run = self._runs.get(response_id)
            return self._copy(run) if run is not None else None

    async def find(self, idempotency_digest: str) -> StoredRun | None:
        """Read the run holding one idempotency digest."""
        async with self._lock:
            response_id = self._idempotency.get(idempotency_digest)
            return self._copy(self._runs[response_id]) if response_id else None

    async def mark_in_progress(
        self,
        response_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Move a queued run in progress."""
        async with self._lock:
            run = self._runs.get(response_id)
            if run is None or run.terminal:
                return None
            return self._put(
                run.model_copy(
                    update={
                        "status": ResponseStatus.IN_PROGRESS,
                        "response": {
                            **run.response,
                            "status": ResponseStatus.IN_PROGRESS.value,
                        },
                        "updated_at": now,
                    }
                )
            )

    async def finish(
        self,
        response_id: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
    ) -> StoredRun | None:
        """Commit a terminal Response unless another terminal outcome won."""
        async with self._lock:
            run = self._runs.get(response_id)
            if run is None or run.terminal:
                return self._copy(run) if run is not None else None
            return self._put(
                run.model_copy(
                    update={
                        "response": response,
                        "status": terminal_status(response),
                        "result_expires_at": now + result_retention,
                        "cleanup_pending": True,
                        "updated_at": now,
                    }
                )
            )

    async def claim_queued(
        self,
        *,
        created_before: datetime,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through runs still queued since before a time."""
        async with self._lock:
            return self._claim(
                (
                    run
                    for run in self._runs.values()
                    if run.status is ResponseStatus.QUEUED
                    and run.created_at < created_before
                ),
                now=now,
                limit=limit,
            )

    async def claim_cleanup_ready(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through terminal runs awaiting checkpoint deletion."""
        async with self._lock:
            return self._claim(
                (run for run in self._runs.values() if run.cleanup_pending),
                now=now,
                limit=limit,
            )

    async def finish_cleanup(self, response_id: str, *, now: datetime) -> None:
        """Record that checkpoint cleanup is done or cannot be performed."""
        async with self._lock:
            run = self._runs.get(response_id)
            if run is not None:
                self._put(
                    run.model_copy(update={"cleanup_pending": False, "updated_at": now})
                )

    async def expire(self, *, now: datetime, limit: int) -> int:
        """Delete expired runs without pending cleanup."""
        async with self._lock:
            expired = [
                run.response_id
                for run in self._runs.values()
                if run.result_expires_at is not None
                and run.result_expires_at <= now
                and not run.cleanup_pending
            ][:limit]
            for response_id in expired:
                digest = self._runs.pop(response_id).idempotency_digest
                if digest is not None:
                    del self._idempotency[digest]
            return len(expired)

    def _claim(
        self,
        runs: Iterable[StoredRun],
        *,
        now: datetime,
        limit: int,
    ) -> list[StoredRun]:
        oldest = sorted(runs, key=lambda run: (run.updated_at, run.response_id))
        return [
            self._put(run.model_copy(update={"updated_at": now}))
            for run in oldest[:limit]
        ]

    def _put(self, run: StoredRun) -> StoredRun:
        self._runs[run.response_id] = run
        return self._copy(run)

    @staticmethod
    def _copy(run: StoredRun) -> StoredRun:
        return run.model_copy(deep=True)


class InMemoryBackgroundBackend:
    """Execute background Responses once inside one application process."""

    def __init__(
        self,
        *,
        graphs: GraphRegistry,
        settings: BackgroundSettings | None = None,
        maintenance_interval: timedelta = timedelta(minutes=1),
    ) -> None:
        self.settings = settings or BackgroundSettings()
        self.store = InMemoryResponseStore()
        self.worker = BackgroundWorker(
            graphs=graphs,
            store=self.store,
            settings=self.settings,
        )
        self._maintenance_interval = maintenance_interval
        self._tasks: TaskGroup | None = None
        self._scopes: dict[str, CancelScope] = {}

    @asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncIterator[None]:
        """Own process-local tasks for one ASGI application lifespan."""
        if self._tasks is not None:
            msg = "The in-memory background backend is already running."
            raise RuntimeError(msg)
        async with create_task_group() as tasks:
            self._tasks = tasks
            tasks.start_soon(self._maintain)
            try:
                yield
            finally:
                self._tasks = None
                tasks.cancel_scope.cancel()

    async def submit(self, run: StoredRun) -> None:
        """Start one process-local execution unless the run already has one."""
        if self._tasks is None:
            msg = "The in-memory background backend lifespan is not running."
            raise RuntimeError(msg)
        if run.response_id not in self._scopes:
            self._tasks.start_soon(self._execute, run.response_id)

    async def stop(self, run: StoredRun) -> None:
        """Cancel the run's task; maintenance then deletes its checkpoints."""
        scope = self._scopes.get(run.response_id)
        if scope is not None:
            scope.cancel()

    async def _execute(self, response_id: str) -> None:
        with CancelScope() as scope:
            self._scopes[response_id] = scope
            try:
                await self.worker.execute(response_id)
            except Exception:  # ruff: ignore[blind-except] - Every failure is retryable; without engine retries, finalize at once.
                try:
                    await self.worker.finalize(response_id)
                except Exception:
                    logger.exception(
                        "background.in_memory_execution_failed",
                        extra={"response_id": response_id},
                    )
            finally:
                del self._scopes[response_id]

    async def _maintain(self) -> None:
        while True:
            await sleep(self._maintenance_interval.total_seconds())
            try:
                await self.worker.maintain(resubmit=self.submit)
            except Exception:
                logger.exception("background.in_memory_maintenance_failed")


__all__ = ["InMemoryBackgroundBackend", "InMemoryResponseStore"]
