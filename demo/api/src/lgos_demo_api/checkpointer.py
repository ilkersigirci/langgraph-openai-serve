"""PostgreSQL persistence wiring for the demo API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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

    checkpointer: AsyncPostgresSaver
    run_coordinator: PostgresRunCoordinator


@asynccontextmanager
async def postgres_runtime(postgres_uri: str) -> AsyncIterator[PostgresRuntime]:
    """Open one ready pool for checkpointing and run coordination."""
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
            run_coordinator=PostgresRunCoordinator(
                pool,
                max_concurrent_leases=_MAX_COORDINATION_LEASES,
            ),
        )


async def setup_postgres_schema(postgres_uri: str) -> None:
    """Initialize or migrate the checkpoint schema once before workers start."""
    async with AsyncPostgresSaver.from_conn_string(postgres_uri) as checkpointer:
        await checkpointer.setup()


__all__ = [
    "PostgresRuntime",
    "postgres_runtime",
    "setup_postgres_schema",
]
