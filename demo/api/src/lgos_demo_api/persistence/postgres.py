"""PostgreSQL persistence wiring for the demo API and background worker."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import anyio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph_openai_serve.integrations.postgres import PostgresRunCoordinator
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 5
_MAX_COORDINATION_LEASES = _POOL_MAX_SIZE - 1
_SCHEMA_LOCK_ID = 0x4C474F535343484D

logger = logging.getLogger(__name__)

PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class PostgresRuntime:
    """Process-local graph dependencies backed by one PostgreSQL pool."""

    checkpointer: AsyncPostgresSaver
    store: AsyncPostgresStore
    run_coordinator: PostgresRunCoordinator


@asynccontextmanager
async def postgres_runtime(postgres_uri: str) -> AsyncGenerator[PostgresRuntime, None]:
    """
    Open one ready pool for checkpoints, Store data, and interrupt coordination.

    Yields:
        Configured PostgreSQL-backed graph dependencies.

    """
    await setup_postgres_schema(postgres_uri)
    pool_context = cast(
        "PostgresPool",
        AsyncConnectionPool(
            conninfo=postgres_uri,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            min_size=_POOL_MIN_SIZE,
            max_size=_POOL_MAX_SIZE,
            open=False,
        ),
    )
    async with pool_context as pool:
        await pool.wait()
        yield PostgresRuntime(
            checkpointer=AsyncPostgresSaver(pool),
            store=AsyncPostgresStore(pool),
            run_coordinator=PostgresRunCoordinator(
                pool,
                max_concurrent_leases=_MAX_COORDINATION_LEASES,
            ),
        )


async def setup_postgres_schema(postgres_uri: str) -> None:
    """Apply pending LangGraph migrations before this process serves work."""
    logger.info("demo.persistence_schema.initializing")
    async with await AsyncConnection[dict[str, Any]].connect(
        postgres_uri, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as connection:
        # A blocking advisory-lock query holds a snapshot that can deadlock with
        # LangGraph's CREATE INDEX CONCURRENTLY. Poll in autocommit instead, so
        # waiting replicas leave no active snapshot. Closing this dedicated
        # session releases the lock on success or failure.
        with anyio.fail_after(60):
            while True:
                cursor = await connection.execute(
                    "SELECT pg_try_advisory_lock(%s) AS acquired", (_SCHEMA_LOCK_ID,)
                )
                row = await cursor.fetchone()
                assert row is not None  # ruff: ignore[assert] - SELECT pg_try_advisory_lock always returns one row.
                if row["acquired"]:
                    break
                await anyio.sleep(0.1)
        await AsyncPostgresSaver(connection).setup()
        await AsyncPostgresStore(connection).setup()
    logger.info("demo.persistence_schema.ready")


__all__ = [
    "PostgresRuntime",
    "postgres_runtime",
    "setup_postgres_schema",
]
