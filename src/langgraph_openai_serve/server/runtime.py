"""Persistence the server owns and passes to the application's registry."""

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import anyio
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import (
    InMemoryRunCoordinator,
    RunCoordinator,
    delete_expired_interrupt_runs,
)
from langgraph_openai_serve.integrations.postgres import PostgresRunCoordinator
from langgraph_openai_serve.server.settings import ServerSettings

logger = get_logger(__name__)

_SCHEMA_LOCK_ID = 0x4C474F535343484D

PostgresPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ServerResources:
    """Persistence shared by every graph one server process runs."""

    checkpointer: BaseCheckpointSaver
    store: BaseStore
    run_coordinator: RunCoordinator


RegistryFactory = Callable[
    [ServerResources],
    GraphRegistry | AbstractAsyncContextManager[GraphRegistry],
]
"""Build the served graphs, or open them for the process lifetime."""


@asynccontextmanager
async def open_resources(
    settings: ServerSettings,
) -> AsyncGenerator[ServerResources, None]:
    """
    Open PostgreSQL persistence, or process memory when no database is set.

    Yields:
        The checkpointer, store, and run coordinator for this process.

    """
    if settings.POSTGRES_URI is None:
        logger.warning("server.persistence.in_memory")
        yield ServerResources(
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
            run_coordinator=InMemoryRunCoordinator(),
        )
        return

    postgres_uri = settings.POSTGRES_URI.get_secret_value()
    await _setup_schema(postgres_uri)
    pool_context = cast(
        "PostgresPool",
        AsyncConnectionPool(
            conninfo=postgres_uri,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            min_size=1,
            max_size=settings.POSTGRES_POOL_SIZE,
            open=False,
        ),
    )
    async with pool_context as pool:
        await pool.wait()
        yield ServerResources(
            checkpointer=AsyncPostgresSaver(pool),
            store=AsyncPostgresStore(pool),
            # Reserve one connection for checkpoints while leases hold the others.
            run_coordinator=PostgresRunCoordinator(
                pool, max_concurrent_leases=settings.POSTGRES_POOL_SIZE - 1
            ),
        )


@asynccontextmanager
async def open_registry(
    factory: RegistryFactory,
    resources: ServerResources,
) -> AsyncGenerator[GraphRegistry, None]:
    """
    Build the registry, entering it when the factory owns resources.

    Yields:
        The registry to serve until the process stops.

    """
    registry = factory(resources)
    if isinstance(registry, GraphRegistry):
        yield registry
        return
    async with registry as opened:
        yield opened


async def expire_paused_runs(
    resources: ServerResources,
    settings: ServerSettings,
) -> None:
    """Delete abandoned interrupt runs on an interval, like Agent Server's TTL."""
    ttl = timedelta(minutes=settings.INTERRUPT_TTL_MINUTES)
    while True:
        try:
            deleted = await delete_expired_interrupt_runs(
                resources.checkpointer, resources.run_coordinator, older_than=ttl
            )
        except Exception:
            logger.exception("server.interrupt_expiry.failed")
        else:
            if deleted:
                logger.info("server.interrupt_expiry.deleted", extra={"runs": deleted})
        await anyio.sleep(settings.INTERRUPT_SWEEP_INTERVAL_MINUTES * 60)


async def _setup_schema(postgres_uri: str) -> None:
    """Apply pending LangGraph migrations before this process serves work."""
    async with await AsyncConnection[dict[str, Any]].connect(
        postgres_uri, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as connection:
        # A blocking advisory-lock query holds a snapshot that can deadlock with
        # LangGraph's CREATE INDEX CONCURRENTLY across replicas. Poll in
        # autocommit instead; closing this session releases the lock.
        with anyio.fail_after(60):
            while True:
                cursor = await connection.execute(
                    "SELECT pg_try_advisory_lock(%s) AS acquired", (_SCHEMA_LOCK_ID,)
                )
                row = await cursor.fetchone()
                if row is not None and row["acquired"]:
                    break
                await anyio.sleep(0.1)
        await AsyncPostgresSaver(connection).setup()
        await AsyncPostgresStore(connection).setup()
