"""PostgreSQL persistence for background Response state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, LiteralString

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from langgraph_openai_serve.background.store import (
    NewRun,
    StoredRun,
    terminal_status,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime, timedelta

    from pydantic import JsonValue

# Serializes concurrent setup calls from several processes.
_SETUP_LOCK = 5_494_716_043_740_046_884

# Append-only, like LangGraph's AsyncPostgresSaver.MIGRATIONS: setup() applies
# every entry after the highest recorded version. Never edit a released entry.
MIGRATIONS: tuple[LiteralString, ...] = (
    """CREATE TABLE IF NOT EXISTS lgos_background_migrations (
    v INTEGER PRIMARY KEY
)""",
    """CREATE TABLE lgos_background_responses (
    response_id text PRIMARY KEY,
    owner_scope text NOT NULL,
    model text NOT NULL,
    checkpoint_thread_id text NOT NULL,
    envelope jsonb NOT NULL,
    response jsonb NOT NULL,
    status text NOT NULL CHECK (
        status IN (
            'queued', 'in_progress', 'completed', 'incomplete', 'failed', 'cancelled'
        )
    ),
    prior_ids text[] NOT NULL,
    idempotency_digest text,
    request_fingerprint text,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    result_expires_at timestamptz,
    cleanup_pending boolean NOT NULL
)""",
    """CREATE UNIQUE INDEX lgos_background_idempotency
    ON lgos_background_responses (idempotency_digest)
    WHERE idempotency_digest IS NOT NULL""",
    """CREATE INDEX lgos_background_queued
    ON lgos_background_responses (updated_at, response_id)
    WHERE status = 'queued'""",
    """CREATE INDEX lgos_background_cleanup
    ON lgos_background_responses (updated_at, response_id)
    WHERE cleanup_pending""",
    """CREATE INDEX lgos_background_expiry
    ON lgos_background_responses (result_expires_at)
    WHERE NOT cleanup_pending""",
)

_PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]


class PostgresResponseStore:
    """
    Persist Response runs with single-statement atomic transitions.

    The pool must return mapping rows (``row_factory=dict_row``), as
    ``AsyncPostgresSaver`` also requires when both share one pool.
    """

    def __init__(self, pool: _PostgresPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """Apply pending schema migrations."""
        async with (
            self._pool.connection() as connection,
            connection.transaction(),
        ):
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (_SETUP_LOCK,))
            await connection.execute(MIGRATIONS[0])
            cursor = await connection.execute(
                "SELECT max(v) AS v FROM lgos_background_migrations"
            )
            row = await cursor.fetchone()
            applied = row["v"] if row is not None and row["v"] is not None else 0
            for version in range(applied + 1, len(MIGRATIONS)):
                await connection.execute(MIGRATIONS[version])
                await connection.execute(
                    "INSERT INTO lgos_background_migrations (v) VALUES (%s)",
                    (version,),
                )

    async def create(self, run: NewRun) -> StoredRun:
        """Persist a new queued run, or return the run holding its digest."""
        inserted = await self._one(
            "INSERT INTO lgos_background_responses ("
            "response_id, owner_scope, model, checkpoint_thread_id, "
            "envelope, response, status, prior_ids, "
            "idempotency_digest, request_fingerprint, "
            "created_at, updated_at, cleanup_pending"
            ") VALUES ("
            "%s, %s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s, %s, false"
            ") ON CONFLICT (idempotency_digest) WHERE idempotency_digest IS NOT NULL "
            "DO NOTHING RETURNING *",
            (
                run.response_id,
                run.owner_scope,
                run.model,
                run.checkpoint_thread_id,
                Jsonb(run.envelope),
                Jsonb(run.response),
                list(run.prior_ids),
                run.idempotency_digest,
                run.request_fingerprint,
                run.created_at,
                run.created_at,
            ),
        )
        if inserted is not None:
            return inserted
        existing = (
            await self.find(run.idempotency_digest)
            if run.idempotency_digest is not None
            else None
        )
        if existing is None:
            # The conflicting run expired between both statements.
            msg = "The idempotent background run expired during creation; retry."
            raise RuntimeError(msg)
        return existing

    async def get(self, response_id: str) -> StoredRun | None:
        """Read one run."""
        return await self._one(
            "SELECT * FROM lgos_background_responses WHERE response_id = %s",
            (response_id,),
        )

    async def find(self, idempotency_digest: str) -> StoredRun | None:
        """Read the run holding one idempotency digest."""
        return await self._one(
            "SELECT * FROM lgos_background_responses WHERE idempotency_digest = %s",
            (idempotency_digest,),
        )

    async def mark_in_progress(
        self,
        response_id: str,
        *,
        now: datetime,
    ) -> StoredRun | None:
        """Move a queued run in progress."""
        return await self._one(
            "UPDATE lgos_background_responses SET status = 'in_progress', "
            "response = jsonb_set(response, '{status}', to_jsonb('in_progress'::text)), "
            "updated_at = %s "
            "WHERE response_id = %s AND status IN ('queued', 'in_progress') "
            "RETURNING *",
            (now, response_id),
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
        finished = await self._one(
            "UPDATE lgos_background_responses SET status = %s, response = %s, "
            "result_expires_at = %s, cleanup_pending = true, updated_at = %s "
            "WHERE response_id = %s AND status IN ('queued', 'in_progress') "
            "RETURNING *",
            (
                terminal_status(response).value,
                Jsonb(response),
                now + result_retention,
                now,
                response_id,
            ),
        )
        # A terminal row never changes status again, so a plain read returns
        # the stable winner when the guarded update matched nothing.
        return finished or await self.get(response_id)

    async def claim_queued(
        self,
        *,
        created_before: datetime,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through runs still queued since before a time."""
        return await self._all(
            "UPDATE lgos_background_responses SET updated_at = %s "
            "WHERE response_id IN ("
            "SELECT response_id FROM lgos_background_responses "
            "WHERE status = 'queued' AND created_at < %s "
            "ORDER BY updated_at, response_id "
            "LIMIT %s FOR UPDATE SKIP LOCKED"
            ") RETURNING *",
            (now, created_before, limit),
        )

    async def claim_cleanup_ready(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> Sequence[StoredRun]:
        """Rotate through terminal runs awaiting checkpoint deletion."""
        return await self._all(
            "UPDATE lgos_background_responses SET updated_at = %s "
            "WHERE response_id IN ("
            "SELECT response_id FROM lgos_background_responses "
            "WHERE cleanup_pending ORDER BY updated_at, response_id "
            "LIMIT %s FOR UPDATE SKIP LOCKED"
            ") RETURNING *",
            (now, limit),
        )

    async def finish_cleanup(self, response_id: str, *, now: datetime) -> None:
        """Record that checkpoint cleanup is done or cannot be performed."""
        async with self._pool.connection() as connection:
            await connection.execute(
                "UPDATE lgos_background_responses "
                "SET cleanup_pending = false, updated_at = %s WHERE response_id = %s",
                (now, response_id),
            )

    async def expire(self, *, now: datetime, limit: int) -> int:
        """Delete expired runs without pending cleanup."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM lgos_background_responses WHERE response_id IN ("
                "SELECT response_id FROM lgos_background_responses "
                "WHERE NOT cleanup_pending AND result_expires_at <= %s "
                "LIMIT %s FOR UPDATE SKIP LOCKED"
                ")",
                (now, limit),
            )
            return cursor.rowcount

    async def _one(
        self,
        statement: LiteralString,
        params: tuple[object, ...],
    ) -> StoredRun | None:
        runs = await self._all(statement, params)
        return runs[0] if runs else None

    async def _all(
        self,
        statement: LiteralString,
        params: tuple[object, ...],
    ) -> list[StoredRun]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(statement, params)
            return [StoredRun.model_validate(row) for row in await cursor.fetchall()]


__all__ = ["MIGRATIONS", "PostgresResponseStore"]
