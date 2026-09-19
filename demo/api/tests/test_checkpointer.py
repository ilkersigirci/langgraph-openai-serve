from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from lgos_demo_api import checkpointer as demo_checkpointer

POSTGRES_URI = "postgresql://example"


async def test_postgres_runtime_owns_one_ready_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = AsyncMock(name="pool", wait=AsyncMock())
    pool.__aenter__.return_value = pool
    saver = Mock(name="saver")
    store = Mock(name="store")
    coordinator = Mock(name="coordinator")
    response_store = Mock(name="response_store")
    pool_factory = Mock(return_value=pool)
    saver_factory = Mock(return_value=saver)
    store_factory = Mock(return_value=store)
    coordinator_factory = Mock(return_value=coordinator)
    response_store_factory = Mock(return_value=response_store)
    monkeypatch.setattr(demo_checkpointer, "AsyncConnectionPool", pool_factory)
    monkeypatch.setattr(demo_checkpointer, "AsyncPostgresSaver", saver_factory)
    monkeypatch.setattr(demo_checkpointer, "AsyncPostgresStore", store_factory)
    monkeypatch.setattr(
        demo_checkpointer,
        "PostgresRunCoordinator",
        coordinator_factory,
    )
    monkeypatch.setattr(
        demo_checkpointer,
        "PostgresResponseStore",
        response_store_factory,
    )

    async with demo_checkpointer.postgres_runtime(POSTGRES_URI) as runtime:
        assert runtime.checkpointer is saver
        assert runtime.store is store
        assert runtime.run_coordinator is coordinator
        assert runtime.response_store is response_store
        pool.__aenter__.assert_awaited_once_with()
        pool.wait.assert_awaited_once_with()
        pool.__aexit__.assert_not_awaited()

    pool_factory.assert_called_once_with(
        conninfo=POSTGRES_URI,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": demo_checkpointer.dict_row,
        },
        min_size=1,
        max_size=5,
        open=False,
    )
    saver_factory.assert_called_once_with(pool)
    store_factory.assert_called_once_with(pool)
    coordinator_factory.assert_called_once_with(
        pool,
        max_concurrent_leases=4,
    )
    response_store_factory.assert_called_once_with(pool)
    pool.__aexit__.assert_awaited_once_with(None, None, None)


async def test_setup_postgres_schema_runs_langgraph_setups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = Mock(setup=AsyncMock())
    store = Mock(setup=AsyncMock())
    response_store = Mock(setup=AsyncMock())
    runtime = Mock(
        checkpointer=saver,
        store=store,
        response_store=response_store,
    )

    @asynccontextmanager
    async def postgres_runtime(postgres_uri: str):
        assert postgres_uri == POSTGRES_URI
        yield runtime

    runtime_factory = Mock(wraps=postgres_runtime)
    monkeypatch.setattr(demo_checkpointer, "postgres_runtime", runtime_factory)

    await demo_checkpointer.setup_postgres_schema(POSTGRES_URI)

    runtime_factory.assert_called_once_with(POSTGRES_URI)
    saver.setup.assert_awaited_once_with()
    store.setup.assert_awaited_once_with()
    response_store.setup.assert_awaited_once_with()
