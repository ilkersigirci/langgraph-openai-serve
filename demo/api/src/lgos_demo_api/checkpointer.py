"""PostgreSQL checkpointer lifecycle helpers for the demo API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


@asynccontextmanager
async def postgres_checkpointer(
    postgres_uri: str,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Create a checkpointer backed by a process-local connection pool."""
    pool_context = cast(
        "AsyncConnectionPool[AsyncConnection[dict[str, Any]]]",
        AsyncConnectionPool(
            conninfo=postgres_uri,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            min_size=1,
            max_size=5,
            open=False,
        ),
    )
    async with pool_context as pool:
        yield AsyncPostgresSaver(pool)


async def setup_postgres_schema(postgres_uri: str) -> None:
    """Initialize or migrate the checkpoint schema once before workers start."""
    async with postgres_checkpointer(postgres_uri) as checkpointer:
        await checkpointer.setup()
