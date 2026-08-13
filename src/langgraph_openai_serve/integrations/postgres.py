"""PostgreSQL coordination for interrupt-enabled graph runs."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from threading import BoundedSemaphore
from typing import Any

from anyio import CancelScope
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from langgraph_openai_serve.graph.interrupt.coordination import RunBusyError

_TRY_ADVISORY_LOCK_SQL = "SELECT pg_try_advisory_lock(%s) AS acquired"
_UNLOCK_ADVISORY_LOCK_SQL = "SELECT pg_advisory_unlock(%s) AS released"

_PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]
logger = logging.getLogger(__name__)


class PostgresRunCoordinator:
    """Coordinate runs with PostgreSQL session advisory locks.

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
        if getattr(pool, "close_returns", False) is True:
            raise ValueError(
                "PostgresRunCoordinator requires a pool with close_returns=False."
            )
        if (
            isinstance(max_concurrent_leases, bool)
            or not isinstance(max_concurrent_leases, int)
            or max_concurrent_leases < 1
        ):
            raise ValueError("max_concurrent_leases must be a positive integer")
        self._pool = pool
        self._capacity = BoundedSemaphore(max_concurrent_leases)

    @asynccontextmanager
    async def __call__(self, key: str, /) -> AsyncIterator[None]:
        if not self._capacity.acquire(blocking=False):
            raise RunBusyError(key)
        try:
            lock_key = _advisory_lock_key(key)
            async with self._pool.connection() as connection:
                if not await _try_acquire_advisory_lock(connection, lock_key):
                    raise RunBusyError(key)

                body_error: BaseException | None = None
                try:
                    yield
                except BaseException as exc:
                    body_error = exc
                    raise
                finally:
                    try:
                        await _release_advisory_lock(connection, lock_key)
                    except Exception:
                        if body_error is None:
                            raise
                        logger.exception(
                            "Could not release a PostgreSQL graph-run lease; "
                            "the unsafe session was discarded."
                        )
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
        cursor = await connection.execute(
            _TRY_ADVISORY_LOCK_SQL,
            (lock_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(
                "PostgreSQL advisory lease acquisition returned no result."
            )
        return bool(row["acquired"])
    except BaseException:
        # Cancellation may arrive after PostgreSQL acquired the session lock.
        # Closing is the only safe way to resolve an indeterminate result.
        await _discard_connection(connection)
        raise


async def _release_advisory_lock(
    connection: AsyncConnection[dict[str, Any]],
    lock_key: int,
) -> None:
    """Release one session lock or discard the unsafe pooled session."""
    try:
        cursor = await connection.execute(
            _UNLOCK_ADVISORY_LOCK_SQL,
            (lock_key,),
        )
        row = await cursor.fetchone()
        if row is None or not row["released"]:
            raise RuntimeError("PostgreSQL advisory lease could not be released.")
    except BaseException:
        # Session locks survive transaction rollback. Closing makes PostgreSQL
        # release the lock and tells psycopg_pool to replace this connection.
        await _discard_connection(connection)
        raise


async def _discard_connection(
    connection: AsyncConnection[dict[str, Any]],
) -> None:
    """Close an unsafe session even when its request is being cancelled."""
    with CancelScope(shield=True):
        await connection.close()


__all__ = ["PostgresRunCoordinator"]
