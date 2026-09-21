"""PostgreSQL coordination for checkpointed graph runs."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from hashlib import sha256
from threading import BoundedSemaphore
from typing import Any

from anyio import CancelScope, get_cancelled_exc_class, sleep
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.interrupt.coordination import RunBusyError, RunLease

_TRY_ADVISORY_LOCK_SQL = "SELECT pg_try_advisory_lock(%s) AS acquired"
_UNLOCK_ADVISORY_LOCK_SQL = "SELECT pg_advisory_unlock(%s) AS released"
_MONITOR_ADVISORY_LOCK_SQL = "SELECT 1 AS healthy"

_PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]
logger = get_logger(__name__)


class PostgresRunCoordinator:
    """
    Coordinate graph runs with PostgreSQL session advisory locks.

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
        monitor_interval: timedelta = timedelta(seconds=5),
    ) -> None:
        if getattr(pool, "close_returns", False) is True:
            msg = "PostgresRunCoordinator requires a pool with close_returns=False."
            raise ValueError(msg)
        if (
            isinstance(max_concurrent_leases, bool)
            or not isinstance(max_concurrent_leases, int)
            or max_concurrent_leases < 1
        ):
            msg = "max_concurrent_leases must be a positive integer"
            raise ValueError(msg)
        if monitor_interval <= timedelta(0):
            msg = "monitor_interval must be positive"
            raise ValueError(msg)
        self._pool = pool
        self._capacity = BoundedSemaphore(max_concurrent_leases)
        self._monitor_interval = monitor_interval

    @asynccontextmanager
    async def __call__(self, key: str, /) -> AsyncIterator[RunLease]:
        """
        Acquire a PostgreSQL advisory lease for one graph run.

        Yields:
            State that records loss of the exact PostgreSQL lock session.

        """
        if not self._capacity.acquire(blocking=False):
            raise RunBusyError(key)
        try:
            lock_key = _advisory_lock_key(key)
            async with self._pool.connection() as connection:
                if not await _try_acquire_advisory_lock(connection, lock_key):
                    raise RunBusyError(key)

                body_error: BaseException | None = None
                lease = RunLease()
                monitor_errors: list[BaseException] = []
                owner = asyncio.current_task()
                if owner is None:
                    msg = "PostgreSQL lease monitoring requires an asyncio task."
                    raise RuntimeError(msg)
                monitor = asyncio.create_task(
                    _monitor_advisory_lock_session(
                        connection,
                        self._monitor_interval.total_seconds(),
                        owner,
                        lease,
                        monitor_errors,
                    )
                )
                try:
                    yield lease
                except get_cancelled_exc_class() as exc:
                    if monitor_errors:
                        body_error = monitor_errors[0]
                        raise body_error from exc
                    body_error = exc
                    raise
                except BaseException as exc:
                    body_error = exc
                    raise
                finally:
                    monitor.cancel()
                    with suppress(asyncio.CancelledError):
                        await monitor
                    try:
                        await _release_advisory_lock(connection, lock_key)
                    except Exception:
                        if body_error is None:
                            raise
                        logger.exception("postgres.graph_run_lease_release_failed")
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
            msg = "PostgreSQL advisory lease acquisition returned no result."
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
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
            msg = "PostgreSQL advisory lease could not be released."
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
    except BaseException:
        # Session locks survive transaction rollback. Closing makes PostgreSQL
        # release the lock and tells psycopg_pool to replace this connection.
        await _discard_connection(connection)
        raise


async def _monitor_advisory_lock_session(
    connection: AsyncConnection[dict[str, Any]],
    interval: float,
    owner: asyncio.Task[Any],
    lease: RunLease,
    errors: list[BaseException],
) -> None:
    """Fail the owning cancel scope when its exact lock session is lost."""
    try:
        while True:
            await sleep(interval)
            await _check_advisory_lock_session(connection)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # ruff: ignore[blind-except] - Any monitor failure means the exact advisory-lock session is unsafe.
        lease.lost = True
        errors.append(exc)
        owner.cancel()


async def _check_advisory_lock_session(
    connection: AsyncConnection[dict[str, Any]],
) -> None:
    cursor = await connection.execute(_MONITOR_ADVISORY_LOCK_SQL)
    row = await cursor.fetchone()
    if row is None or not row["healthy"]:
        msg = "PostgreSQL advisory-lock session health check failed."
        raise RuntimeError(msg)


async def _discard_connection(
    connection: AsyncConnection[dict[str, Any]],
) -> None:
    """Close an unsafe session even when its request is being cancelled."""
    with CancelScope(shield=True):
        await connection.close()


__all__ = ["PostgresRunCoordinator"]
