"""Single-process background execution for development and examples."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from anyio import CancelScope, Lock, create_task_group, get_cancelled_exc_class

from langgraph_openai_serve.background.contracts import (
    BackgroundSettings,
    RetryableJobError,
    RunJob,
)
from langgraph_openai_serve.background.store import (
    Acceptance,
    BackgroundCapacityError,
    BackgroundIdempotencyConflictError,
    BackgroundResponseExpiredError,
    NewRun,
    ResponseStatus,
    StoredRun,
    expired_run_deletable,
    terminal_run,
    tombstone_run,
)
from langgraph_openai_serve.background.worker import BackgroundWorker
from langgraph_openai_serve.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from anyio.abc import TaskGroup
    from pydantic import JsonValue

    from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = get_logger(__name__)


class InMemoryResponseStore:
    """Keep Response state in one process until it exits."""

    def __init__(self) -> None:
        self._runs: dict[str, StoredRun] = {}
        self._response_ids: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}
        self._lock = Lock()

    async def accept(self, run: NewRun, *, capacity: int) -> Acceptance:
        """Atomically enforce capacity and optional create idempotency."""
        async with self._lock:
            key = self._key(run)
            if key is not None and (
                existing := self._runs.get(self._idempotency.get(key, ""))
            ):
                if (
                    existing.idempotency_expires_at is not None
                    and existing.idempotency_expires_at <= run.created_at
                ):
                    self._idempotency.pop(key, None)
                    self._put(
                        existing.model_copy(
                            update={
                                "idempotency_key": None,
                                "idempotency_expires_at": None,
                                "updated_at": run.created_at,
                                "version": existing.version + 1,
                            }
                        )
                    )
                elif existing.request_fingerprint != run.request_fingerprint:
                    raise BackgroundIdempotencyConflictError
                elif existing.response is None or (
                    existing.result_expires_at is not None
                    and existing.result_expires_at <= run.created_at
                ):
                    raise BackgroundResponseExpiredError
                else:
                    return Acceptance(run=self._copy(existing), created=False)
            if sum(not item.terminal for item in self._runs.values()) >= capacity:
                raise BackgroundCapacityError
            stored = StoredRun(
                **run.model_dump(),
                status=ResponseStatus.QUEUED,
                updated_at=run.created_at,
            )
            self._put(stored)
            self._response_ids[stored.response_id] = stored.run_id
            if key is not None:
                self._idempotency[key] = stored.run_id
            return Acceptance(run=self._copy(stored), created=True)

    async def record_workflow_run(
        self,
        run_id: str,
        workflow_run_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Store an idempotent process-local task receipt."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if run.workflow_run_id is not None:
                return (
                    self._copy(run) if run.workflow_run_id == workflow_run_id else None
                )
            return self._put(
                run.model_copy(
                    update={
                        "workflow_run_id": workflow_run_id,
                        "updated_at": now,
                        "version": run.version + 1,
                    }
                )
            )

    async def discard_unsubmitted(self, run_id: str) -> bool:
        """Remove only an active row without a process-local task receipt."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.terminal or run.workflow_run_id is not None:
                return False
            self._remove(run)
            return True

    async def get(
        self,
        response_id: str,
        owner_scope: str,
        *,
        now: datetime | None = None,
    ) -> StoredRun | None:
        """Read one authorized, unexpired public Response snapshot."""
        checked_at = now or datetime.now(UTC)
        async with self._lock:
            run = self._runs.get(self._response_ids.get(response_id, ""))
            if run is None or run.owner_scope != owner_scope or run.response is None:
                return None
            if (
                run.terminal
                and run.result_expires_at is not None
                and run.result_expires_at <= checked_at
            ):
                return None
            return self._copy(run)

    async def get_internal(self, run_id: str) -> StoredRun | None:
        """Read one trusted run snapshot."""
        async with self._lock:
            run = self._runs.get(run_id)
            return self._copy(run) if run is not None else None

    async def mark_in_progress(
        self,
        run_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Mark an active queued Response in progress."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.terminal:
                return None
            if run.status is ResponseStatus.IN_PROGRESS:
                return self._copy(run)
            response = (
                {**run.response, "status": ResponseStatus.IN_PROGRESS.value}
                if run.response is not None
                else None
            )
            return self._put(
                run.model_copy(
                    update={
                        "status": ResponseStatus.IN_PROGRESS,
                        "response": response,
                        "updated_at": now,
                        "version": run.version + 1,
                    }
                )
            )

    async def request_cancellation(  # ruff: ignore[too-many-arguments] - Mirrors the atomic store contract.
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
        idempotency_retention: timedelta,
    ) -> StoredRun | None:
        """Choose cancellation atomically unless a terminal result won."""
        async with self._lock:
            run = self._runs.get(self._response_ids.get(response_id, ""))
            if run is None or run.owner_scope != owner_scope or run.response is None:
                return None
            if run.terminal:
                return self._copy(run)
            return self._put(
                terminal_run(
                    run,
                    response,
                    now=now,
                    result_retention=result_retention,
                    idempotency_retention=idempotency_retention,
                )
            )

    async def publish_terminal(
        self,
        run_id: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
        idempotency_retention: timedelta,
    ) -> StoredRun | None:
        """Commit a terminal snapshot unless another terminal outcome won."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.terminal:
                return None
            return self._put(
                terminal_run(
                    run,
                    response,
                    now=now,
                    result_retention=result_retention,
                    idempotency_retention=idempotency_retention,
                )
            )

    async def claim_cancellations(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through native cancellations awaiting delivery."""
        async with self._lock:
            runs = [
                run
                for run in sorted(
                    self._runs.values(),
                    key=lambda item: (item.updated_at, item.run_id),
                )
                if run.cancellation_pending and run.workflow_run_id is not None
            ][:limit]
            return [self._claim(run, now=now) for run in runs]

    async def finish_cancellation(self, run_id: str, *, now: datetime) -> bool:
        """Record successful process-local cancellation."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or not run.cancellation_pending:
                return False
            self._put(
                run.model_copy(
                    update={
                        "cancellation_pending": False,
                        "updated_at": now,
                        "version": run.version + 1,
                    }
                )
            )
            return True

    async def claim_cleanup_ready(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through terminal checkpoint lineages awaiting deletion."""
        async with self._lock:
            runs = [
                run
                for run in sorted(
                    self._runs.values(),
                    key=lambda item: (item.updated_at, item.run_id),
                )
                if run.terminal and run.cleanup_pending and not run.recovery_cleaned
            ][:limit]
            return [self._claim(run, now=now) for run in runs]

    async def finish_cleanup(self, run_id: str, *, now: datetime) -> bool:
        """Record successful removal of one checkpoint lineage."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or not run.terminal:
                return False
            self._put(
                run.model_copy(
                    update={
                        "cleanup_pending": False,
                        "recovery_cleaned": True,
                        "updated_at": now,
                        "version": run.version + 1,
                    }
                )
            )
            return True

    async def abandon_cleanup(self, run_id: str, *, now: datetime) -> bool:
        """Stop retrying cleanup without claiming checkpoint deletion."""
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or not run.terminal or not run.cleanup_pending:
                return False
            self._put(
                run.model_copy(
                    update={
                        "cleanup_pending": False,
                        "updated_at": now,
                        "version": run.version + 1,
                    }
                )
            )
            return True

    async def expire(self, *, now: datetime, limit: int) -> int:
        """Convert expired results to tombstones, then remove expired keys."""
        async with self._lock:
            actionable = [
                run
                for run in sorted(self._runs.values(), key=lambda item: item.updated_at)
                if run.terminal
                and run.result_expires_at is not None
                and run.result_expires_at <= now
                and (
                    run.response is not None
                    or bool(run.envelope)
                    or expired_run_deletable(run, now=now)
                )
            ][:limit]
            for run in actionable:
                if expired_run_deletable(run, now=now):
                    self._remove(run)
                else:
                    self._put(tombstone_run(run, now=now))
            return len(actionable)

    @staticmethod
    def _key(run: NewRun | StoredRun) -> tuple[str, str, str] | None:
        if run.idempotency_key is None:
            return None
        return (run.owner_scope, run.model, run.idempotency_key)

    def _put(self, run: StoredRun) -> StoredRun:
        self._runs[run.run_id] = run
        return self._copy(run)

    def _claim(self, run: StoredRun, *, now: datetime) -> StoredRun:
        return self._put(
            run.model_copy(
                update={
                    "updated_at": now,
                    "version": run.version + 1,
                }
            )
        )

    def _remove(self, run: StoredRun) -> None:
        self._runs.pop(run.run_id, None)
        self._response_ids.pop(run.response_id, None)
        key = self._key(run)
        if key is not None and self._idempotency.get(key) == run.run_id:
            self._idempotency.pop(key, None)

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
    ) -> None:
        self.settings = settings or BackgroundSettings()
        self.store = InMemoryResponseStore()
        self.worker = BackgroundWorker(
            graphs=graphs,
            store=self.store,
            settings=self.settings,
        )
        self._tasks: TaskGroup | None = None
        self._scopes: dict[str, CancelScope] = {}
        self._scope_lock = Lock()
        self._submission_lock = Lock()

    @asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncIterator[None]:
        """Own process-local tasks for one ASGI application lifespan."""
        if self._tasks is not None:
            msg = "The in-memory background backend is already running."
            raise RuntimeError(msg)
        try:
            async with create_task_group() as tasks:
                self._tasks = tasks
                try:
                    yield
                finally:
                    self._tasks = None
                    tasks.cancel_scope.cancel()
        finally:
            self._scopes.clear()

    async def create(self, run: NewRun) -> StoredRun:
        """Persist and start one process-local execution."""
        async with self._submission_lock:
            tasks = self._tasks
            if tasks is None:
                msg = "The in-memory background backend lifespan is not running."
                raise RuntimeError(msg)
            accepted = await self.store.accept(
                run,
                capacity=self.settings.admission_capacity,
            )
            if accepted.run.workflow_run_id is not None:
                return accepted.run
            with CancelScope(shield=True):
                recorded = await self.store.record_workflow_run(
                    accepted.run.run_id,
                    accepted.run.run_id,
                    now=datetime.now(UTC),
                )
                if recorded is None:
                    msg = "The process-local task receipt could not be persisted."
                    raise RuntimeError(msg)
                tasks.start_soon(self._execute, RunJob(run_id=recorded.run_id))
            return recorded

    async def retrieve(self, response_id: str, owner_scope: str) -> StoredRun | None:
        """Read one authorized Response snapshot."""
        return await self.store.get(response_id, owner_scope)

    async def cancel(
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        stored: bool,
    ) -> StoredRun | None:
        """Choose cancellation, then stop its process-local task."""
        cancelled = await self.store.request_cancellation(
            response_id,
            owner_scope,
            response,
            now=datetime.now(UTC),
            result_retention=self.settings.result_retention_for(stored=stored),
            idempotency_retention=self.settings.idempotency_retention,
        )
        if (
            cancelled is None
            or cancelled.status is not ResponseStatus.CANCELLED
            or not cancelled.cancellation_pending
        ):
            return cancelled
        with CancelScope(shield=True):
            async with self._scope_lock:
                scope = self._scopes.get(cancelled.run_id)
                if scope is not None:
                    scope.cancel()
            await self.store.finish_cancellation(
                cancelled.run_id,
                now=datetime.now(UTC),
            )
        return await self.store.get(response_id, owner_scope)

    async def _execute(self, job: RunJob) -> None:
        scope = CancelScope()
        async with self._scope_lock:
            self._scopes[job.run_id] = scope
        try:
            await self._run_job(job, scope)
        except get_cancelled_exc_class():
            raise
        except Exception:
            logger.exception(
                "background.in_memory_execution_failed",
                extra={"run_id": job.run_id},
            )
        finally:
            with CancelScope(shield=True):
                async with self._scope_lock:
                    if self._scopes.get(job.run_id) is scope:
                        self._scopes.pop(job.run_id)

    async def _run_job(self, job: RunJob, scope: CancelScope) -> None:
        with scope:
            try:
                await self.worker.execute(job)
            except RetryableJobError:
                await self.worker.finalize(job)
        if scope.cancel_called:
            with CancelScope(shield=True):
                await self.worker.execute(job)


__all__ = ["InMemoryBackgroundBackend", "InMemoryResponseStore"]
