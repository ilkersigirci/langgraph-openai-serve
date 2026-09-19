"""PostgreSQL persistence for background Response state."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from typing import TYPE_CHECKING, Any, LiteralString, cast

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

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
    from collections.abc import AsyncIterator, Sequence

    from pydantic import JsonValue

_TABLE = "lgos_background_responses"
_CAPACITY_LOCK = 5_494_716_043_740_046_884

_PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]
_Connection = AsyncConnection[dict[str, Any]]


class PostgresResponseStore:
    """Persist canonical Response snapshots with small atomic transitions."""

    def __init__(self, pool: _PostgresPool) -> None:
        if getattr(pool, "close_returns", False) is True:
            msg = "PostgresResponseStore requires a pool with close_returns=False."
            raise ValueError(msg)
        self._pool = pool

    async def setup(self) -> None:
        """Create the final background Response schema."""
        schema = files("langgraph_openai_serve.integrations").joinpath(
            "background_schema.sql"
        )
        async with self._pool.connection() as connection:
            statement = cast("LiteralString", schema.read_text(encoding="utf-8"))
            await connection.execute(sql.SQL(statement), prepare=False)

    async def accept(self, run: NewRun, *, capacity: int) -> Acceptance:
        """Atomically enforce capacity and optional create idempotency."""
        async with (
            self._pool.connection() as connection,
            connection.transaction(),
        ):
            await connection.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_CAPACITY_LOCK,),
            )
            if run.idempotency_key is not None:
                existing = await self._idempotent_for_update(connection, run)
                if existing is not None:
                    reservation_expired = (
                        existing.idempotency_expires_at is not None
                        and existing.idempotency_expires_at <= run.created_at
                    )
                    if reservation_expired:
                        await self._write(
                            connection,
                            existing.model_copy(
                                update={
                                    "idempotency_key": None,
                                    "idempotency_expires_at": None,
                                    "updated_at": run.created_at,
                                    "version": existing.version + 1,
                                }
                            ),
                        )
                    elif existing.request_fingerprint != run.request_fingerprint:
                        raise BackgroundIdempotencyConflictError
                    elif existing.response is None or (
                        existing.result_expires_at is not None
                        and existing.result_expires_at <= run.created_at
                    ):
                        raise BackgroundResponseExpiredError
                    else:
                        return Acceptance(run=existing, created=False)

            cursor = await connection.execute(
                _table_sql(
                    "SELECT count(*) AS count FROM ("
                    "SELECT 1 FROM {table} "
                    "WHERE status IN ('queued', 'in_progress') LIMIT %s"
                    ") AS active"
                ),
                (capacity,),
            )
            row = await cursor.fetchone()
            if row is None or int(row["count"]) >= capacity:
                raise BackgroundCapacityError

            stored = StoredRun(
                **run.model_dump(),
                status=ResponseStatus.QUEUED,
                updated_at=run.created_at,
            )
            await self._insert(connection, stored)
            return Acceptance(run=stored, created=True)

    async def record_workflow_run(
        self,
        run_id: str,
        workflow_run_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Store an idempotent Hatchet workflow receipt."""
        async with self._locked_run(run_id) as locked:
            connection, run = locked
            if run is None:
                return None
            if run.workflow_run_id is not None:
                return run if run.workflow_run_id == workflow_run_id else None
            updated = run.model_copy(
                update={
                    "workflow_run_id": workflow_run_id,
                    "updated_at": now,
                    "version": run.version + 1,
                }
            )
            await self._write(connection, updated)
            return updated

    async def discard_unsubmitted(self, run_id: str) -> bool:
        """Remove only an active row with no native workflow receipt."""
        async with self._locked_run(run_id) as locked:
            connection, run = locked
            if run is None or run.terminal or run.workflow_run_id is not None:
                return False
            await connection.execute(
                _table_sql("DELETE FROM {table} WHERE run_id = %s"),
                (run_id,),
            )
            return True

    async def get(
        self,
        response_id: str,
        owner_scope: str,
        *,
        now: datetime | None = None,
    ) -> StoredRun | None:
        """Read one authorized public snapshot without a row lock."""
        checked_at = now or datetime.now(UTC)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                _table_sql(
                    "SELECT record FROM {table} "
                    "WHERE response_id = %s AND owner_scope = %s"
                ),
                (response_id, owner_scope),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        run = _stored(row)
        if run.response is None:
            return None
        if (
            run.terminal
            and run.result_expires_at is not None
            and run.result_expires_at <= checked_at
        ):
            return None
        return run

    async def get_internal(self, run_id: str) -> StoredRun | None:
        """Read one trusted run snapshot without a row lock."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                _table_sql("SELECT record FROM {table} WHERE run_id = %s"),
                (run_id,),
            )
            row = await cursor.fetchone()
        return _stored(row) if row is not None else None

    async def mark_in_progress(
        self,
        run_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Mark an active queued Response in progress."""
        async with self._locked_run(run_id) as locked:
            connection, run = locked
            if run is None or run.terminal:
                return None
            if run.status is ResponseStatus.IN_PROGRESS:
                return run
            response = (
                {**run.response, "status": ResponseStatus.IN_PROGRESS.value}
                if run.response is not None
                else None
            )
            updated = run.model_copy(
                update={
                    "status": ResponseStatus.IN_PROGRESS,
                    "response": response,
                    "updated_at": now,
                    "version": run.version + 1,
                }
            )
            await self._write(connection, updated)
            return updated

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
        """Choose cancellation under a row lock unless a terminal result won."""
        async with self._locked_response(response_id, owner_scope) as locked:
            connection, run = locked
            if run is None or run.response is None:
                return None
            if run.terminal:
                return run
            updated = terminal_run(
                run,
                response,
                now=now,
                result_retention=result_retention,
                idempotency_retention=idempotency_retention,
            )
            await self._write(connection, updated)
            return updated

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
        async with self._locked_run(run_id) as locked:
            connection, run = locked
            if run is None or run.terminal:
                return None
            updated = terminal_run(
                run,
                response,
                now=now,
                result_retention=result_retention,
                idempotency_retention=idempotency_retention,
            )
            await self._write(connection, updated)
            return updated

    async def claim_cancellations(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through native cancellations awaiting delivery."""
        return await self._claim_ready(
            "SELECT record FROM {table} "
            "WHERE cancellation_pending AND workflow_run_id IS NOT NULL "
            "ORDER BY updated_at, run_id FOR UPDATE SKIP LOCKED LIMIT %s",
            now=now,
            limit=limit,
        )

    async def finish_cancellation(self, run_id: str, *, now: datetime) -> bool:
        """Record successful delivery of one Hatchet cancellation."""
        async with self._locked_run(run_id) as locked:
            connection, run = locked
            if run is None or not run.cancellation_pending:
                return False
            updated = run.model_copy(
                update={
                    "cancellation_pending": False,
                    "updated_at": now,
                    "version": run.version + 1,
                }
            )
            await self._write(connection, updated)
            return True

    async def claim_cleanup_ready(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through terminal checkpoint lineages awaiting deletion."""
        return await self._claim_ready(
            "SELECT record FROM {table} "
            "WHERE terminal_at IS NOT NULL "
            "AND cleanup_pending AND NOT recovery_cleaned "
            "ORDER BY updated_at, run_id FOR UPDATE SKIP LOCKED LIMIT %s",
            now=now,
            limit=limit,
        )

    async def _claim_ready(
        self,
        statement: LiteralString,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Persist attempt order so one bad row cannot starve later work."""
        async with (
            self._pool.connection() as connection,
            connection.transaction(),
        ):
            cursor = await connection.execute(
                _table_sql(statement),
                (limit,),
            )
            claimed: list[StoredRun] = []
            for row in await cursor.fetchall():
                run = _stored(row)
                updated = run.model_copy(
                    update={
                        "updated_at": now,
                        "version": run.version + 1,
                    }
                )
                await self._write(connection, updated)
                claimed.append(updated)
            return claimed

    async def finish_cleanup(self, run_id: str, *, now: datetime) -> bool:
        """Record checkpoint deletion under the run row lock."""
        async with self._locked_run(run_id) as locked:
            connection, run = locked
            if run is None or not run.terminal:
                return False
            updated = run.model_copy(
                update={
                    "cleanup_pending": False,
                    "recovery_cleaned": True,
                    "updated_at": now,
                    "version": run.version + 1,
                }
            )
            await self._write(connection, updated)
            return True

    async def expire(self, *, now: datetime, limit: int) -> int:
        """Convert expired results to tombstones, then remove expired keys."""
        async with (
            self._pool.connection() as connection,
            connection.transaction(),
        ):
            cursor = await connection.execute(
                _table_sql(
                    "SELECT record FROM {table} "
                    "WHERE terminal_at IS NOT NULL AND result_expires_at <= %s "
                    "AND ("
                    "COALESCE(record -> 'response', 'null'::jsonb) <> 'null'::jsonb "
                    "OR COALESCE(record -> 'envelope', '{{}}'::jsonb) <> '{{}}'::jsonb "
                    "OR (NOT cancellation_pending "
                    "AND (idempotency_key IS NULL OR idempotency_expires_at IS NULL "
                    "OR idempotency_expires_at <= %s) "
                    "AND (recovery_cleaned OR NOT cleanup_pending))) "
                    "ORDER BY result_expires_at FOR UPDATE SKIP LOCKED LIMIT %s"
                ),
                (now, now, limit),
            )
            changed = 0
            for row in await cursor.fetchall():
                run = _stored(row)
                if expired_run_deletable(run, now=now):
                    await connection.execute(
                        _table_sql("DELETE FROM {table} WHERE run_id = %s"),
                        (run.run_id,),
                    )
                else:
                    await self._write(connection, tombstone_run(run, now=now))
                changed += 1
            return changed

    @staticmethod
    async def _idempotent_for_update(
        connection: _Connection,
        run: NewRun,
    ) -> StoredRun | None:
        cursor = await connection.execute(
            _table_sql(
                "SELECT record FROM {table} "
                "WHERE owner_scope = %s AND model = %s AND idempotency_key = %s "
                "FOR UPDATE"
            ),
            (run.owner_scope, run.model, run.idempotency_key),
        )
        row = await cursor.fetchone()
        return _stored(row) if row is not None else None

    @asynccontextmanager
    async def _locked_run(
        self,
        run_id: str,
    ) -> AsyncIterator[tuple[_Connection, StoredRun | None]]:
        async with (
            self._pool.connection() as connection,
            connection.transaction(),
        ):
            cursor = await connection.execute(
                _table_sql("SELECT record FROM {table} WHERE run_id = %s FOR UPDATE"),
                (run_id,),
            )
            row = await cursor.fetchone()
            yield connection, _stored(row) if row is not None else None

    @asynccontextmanager
    async def _locked_response(
        self,
        response_id: str,
        owner_scope: str,
    ) -> AsyncIterator[tuple[_Connection, StoredRun | None]]:
        async with (
            self._pool.connection() as connection,
            connection.transaction(),
        ):
            cursor = await connection.execute(
                _table_sql(
                    "SELECT record FROM {table} "
                    "WHERE response_id = %s AND owner_scope = %s FOR UPDATE"
                ),
                (response_id, owner_scope),
            )
            row = await cursor.fetchone()
            yield connection, _stored(row) if row is not None else None

    @staticmethod
    async def _insert(connection: _Connection, run: StoredRun) -> None:
        columns, values = _indexed_values(run)
        placeholders = ", ".join(["%s"] * len(columns))
        await connection.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(_TABLE),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
                sql.SQL(placeholders),
            ),
            values,
        )

    @staticmethod
    async def _write(connection: _Connection, run: StoredRun) -> None:
        columns, values = _indexed_values(run)
        assignments = sql.SQL(", ").join(
            sql.SQL("{} = %s").format(sql.Identifier(column)) for column in columns[1:]
        )
        await connection.execute(
            sql.SQL("UPDATE {} SET {} WHERE run_id = %s").format(
                sql.Identifier(_TABLE),
                assignments,
            ),
            (*values[1:], run.run_id),
        )


def _table_sql(statement: LiteralString) -> sql.Composed:
    return sql.SQL(statement).format(table=sql.Identifier(_TABLE))


def _indexed_values(run: StoredRun) -> tuple[tuple[str, ...], tuple[object, ...]]:
    columns = (
        "run_id",
        "response_id",
        "owner_scope",
        "model",
        "idempotency_key",
        "request_fingerprint",
        "status",
        "workflow_run_id",
        "terminal_at",
        "result_expires_at",
        "idempotency_expires_at",
        "cancellation_pending",
        "cleanup_pending",
        "recovery_cleaned",
        "updated_at",
        "version",
        "record",
    )
    values: tuple[object, ...] = (
        run.run_id,
        run.response_id,
        run.owner_scope,
        run.model,
        run.idempotency_key,
        run.request_fingerprint,
        run.status.value,
        run.workflow_run_id,
        run.terminal_at,
        run.result_expires_at,
        run.idempotency_expires_at,
        run.cancellation_pending,
        run.cleanup_pending,
        run.recovery_cleaned,
        run.updated_at,
        run.version,
        Jsonb(run.model_dump(mode="json")),
    )
    return columns, values


def _stored(row: dict[str, Any]) -> StoredRun:
    return StoredRun.model_validate(row["record"])


__all__ = ["PostgresResponseStore"]
