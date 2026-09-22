"""PostgreSQL persistence wiring for the demo API and background worker."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph_openai_serve.integrations.background_postgres import (
    PostgresResponseStore,
)
from langgraph_openai_serve.integrations.postgres import PostgresRunCoordinator
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 5
_MAX_COORDINATION_LEASES = _POOL_MAX_SIZE - 1

PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class PostgresRuntime:
    """Process-local graph dependencies backed by one PostgreSQL pool."""

    pool: PostgresPool
    checkpointer: AsyncPostgresSaver
    store: AsyncPostgresStore
    run_coordinator: PostgresRunCoordinator
    response_store: PostgresResponseStore


def _create_postgres_runtime(postgres_uri: str) -> PostgresRuntime:
    """Construct unopened process-owned PostgreSQL dependencies."""
    pool = cast(
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
    return PostgresRuntime(
        pool=pool,
        checkpointer=AsyncPostgresSaver(pool),
        store=AsyncPostgresStore(pool),
        run_coordinator=PostgresRunCoordinator(
            pool,
            max_concurrent_leases=_MAX_COORDINATION_LEASES,
        ),
        response_store=PostgresResponseStore(pool),
    )


@asynccontextmanager
async def open_postgres_runtime(
    runtime: PostgresRuntime,
) -> AsyncIterator[PostgresRuntime]:
    """Open and own one previously constructed PostgreSQL runtime.

    Yields:
        Configured PostgreSQL-backed graph dependencies.
    """
    async with runtime.pool as pool:
        await pool.wait()
        yield runtime


@asynccontextmanager
async def postgres_runtime(postgres_uri: str) -> AsyncIterator[PostgresRuntime]:
    """Construct, open, and own one process-local PostgreSQL runtime."""
    async with open_postgres_runtime(_create_postgres_runtime(postgres_uri)) as runtime:
        yield runtime


async def setup_postgres_schema(postgres_uri: str) -> None:
    """Initialize LangGraph and LGOS background persistence schemas once."""
    async with postgres_runtime(postgres_uri) as runtime:
        await runtime.checkpointer.setup()
        await runtime.store.setup()
        await runtime.response_store.setup()


__all__ = [
    "PostgresRuntime",
    "postgres_runtime",
    "setup_postgres_schema",
]
