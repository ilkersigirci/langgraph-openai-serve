"""Process-local background fakes shared by package tests."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from anyio import Lock

from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background.contracts import BackgroundSettings, RunJob
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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue


class MemoryResponseStore:
    """Deterministic implementation of the store protocol for unit tests."""

    def __init__(self) -> None:
        self._runs: dict[str, StoredRun] = {}
        self._response_ids: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}
        self._lock = Lock()

    async def accept(self, run: NewRun, *, capacity: int) -> Acceptance:
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
        async with self._lock:
            run = self._runs.get(run_id)
            return self._copy(run) if run is not None else None

    async def mark_in_progress(
        self,
        run_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
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

    async def request_cancellation(  # ruff: ignore[too-many-arguments]
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        now: datetime,
        result_retention: timedelta,
        idempotency_retention: timedelta,
    ) -> StoredRun | None:
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

    async def expire(self, *, now: datetime, limit: int) -> int:
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


class MemoryBackgroundBackend:
    """Manually driven lifecycle backend for API tests."""

    def __init__(
        self,
        *,
        store: MemoryResponseStore | None = None,
        settings: BackgroundSettings | None = None,
    ) -> None:
        self.store = store or MemoryResponseStore()
        self.settings = settings or BackgroundSettings()
        self._pending: deque[RunJob] = deque()
        self._cancelled: set[str] = set()
        self._lock = Lock()

    async def create(self, run: NewRun) -> StoredRun:
        accepted = await self.store.accept(
            run,
            capacity=self.settings.admission_capacity,
        )
        if accepted.run.workflow_run_id is not None:
            return accepted.run
        recorded = await self.store.record_workflow_run(
            accepted.run.run_id,
            accepted.run.run_id,
            now=datetime.now(UTC),
        )
        if recorded is None:
            msg = "Test workflow receipt could not be persisted."
            raise RuntimeError(msg)
        async with self._lock:
            self._pending.append(RunJob(run_id=recorded.run_id))
        return recorded

    async def retrieve(self, response_id: str, owner_scope: str) -> StoredRun | None:
        return await self.store.get(response_id, owner_scope)

    async def cancel(
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
    ) -> StoredRun | None:
        current = await self.store.get(response_id, owner_scope)
        if current is None:
            return None
        request = ResponseCreateRequest.model_validate(current.envelope)
        retention = self.settings.result_retention_for(stored=bool(request.store))
        result = await self.store.request_cancellation(
            response_id,
            owner_scope,
            response,
            now=datetime.now(UTC),
            result_retention=retention,
            idempotency_retention=self.settings.idempotency_retention,
        )
        if result is not None and result.status is ResponseStatus.CANCELLED:
            async with self._lock:
                self._cancelled.add(result.run_id)
        return result

    async def receive(self) -> RunJob | None:
        async with self._lock:
            while self._pending:
                job = self._pending.popleft()
                if job.run_id not in self._cancelled:
                    return job
            return None


__all__ = ["MemoryBackgroundBackend", "MemoryResponseStore"]
