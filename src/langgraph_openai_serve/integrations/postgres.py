"""PostgreSQL coordination for interrupt-enabled graph runs."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from hashlib import sha256
from threading import BoundedSemaphore
from typing import Any

from anyio import CancelScope
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from langgraph_openai_serve.graph.interrupt import RunBusyError

_TRY_ADVISORY_LOCK_SQL = "SELECT pg_try_advisory_lock(%s) AS acquired"
_UNLOCK_ADVISORY_LOCK_SQL = "SELECT pg_advisory_unlock(%s) AS released"

_PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]


class PostgresRunCoordinator:
    """
    Coordinate interrupt runs with PostgreSQL session advisory locks.

    The pool must return mapping rows, as required by ``AsyncPostgresSaver``
    when both components share one pool (for example, ``row_factory=dict_row``).
    ``max_concurrent_leases`` limits how many pool connections coordination may
    hold at once. When the checkpointer shares this pool, reserve at least one
    connection for checkpoint I/O to avoid exhausting the pool with leases.
    """

    def __init__(
        self,
        pool: _PostgresPool,
        *,
        max_concurrent_leases: int,
    ) -> None:
        # With close_returns, close() hands the session back to the pool, so a
        # discarded session would keep holding its advisory lock.
        if getattr(pool, "close_returns", False):
            msg = "PostgresRunCoordinator requires a pool with close_returns=False."
            raise ValueError(msg)
        if max_concurrent_leases < 1:
            msg = "max_concurrent_leases must be a positive integer"
            raise ValueError(msg)
        self._pool = pool
        self._capacity = BoundedSemaphore(max_concurrent_leases)

    @asynccontextmanager
    async def __call__(self, key: str, /) -> AsyncGenerator[None, None]:
        """Hold a PostgreSQL advisory lock for one interrupt run."""
        if not self._capacity.acquire(blocking=False):
            raise RunBusyError
        try:
            lock_key = _advisory_lock_key(key)
            async with self._pool.connection() as connection:
                if not await _try_acquire_advisory_lock(connection, lock_key):
                    raise RunBusyError
                try:
                    yield
                finally:
                    await _release_advisory_lock(connection, lock_key)
        finally:
            self._capacity.release()


def _advisory_lock_key(value: str) -> int:
    digest = sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


async def _try_acquire_advisory_lock(
    connection: AsyncConnection[dict[str, Any]],
    lock_key: int,
) -> bool:
    """Acquire one session lock or discard a session with unknown state."""
    try:
        cursor = await connection.execute(_TRY_ADVISORY_LOCK_SQL, (lock_key,))
        row = await cursor.fetchone()
    except BaseException:
        # Cancellation may arrive after PostgreSQL acquired the session lock.
        # Closing is the only safe way to resolve an indeterminate result.
        await _discard_connection(connection)
        raise
    return bool(row and row["acquired"])


async def _release_advisory_lock(
    connection: AsyncConnection[dict[str, Any]],
    lock_key: int,
) -> None:
    """Release one session lock or discard the unsafe pooled session."""
    try:
        cursor = await connection.execute(_UNLOCK_ADVISORY_LOCK_SQL, (lock_key,))
        row = await cursor.fetchone()
    except BaseException:
        # Session locks survive transaction rollback. Closing makes PostgreSQL
        # release the lock and tells psycopg_pool to replace this connection.
        await _discard_connection(connection)
        raise
    if not (row and row["released"]):
        await _discard_connection(connection)
        msg = "PostgreSQL advisory lease could not be released."
        raise RuntimeError(msg)


async def _discard_connection(
    connection: AsyncConnection[dict[str, Any]],
) -> None:
    """Close an unsafe session even when its request is being cancelled."""
    with CancelScope(shield=True):
        await connection.close()


__all__ = ["PostgresRunCoordinator"]
