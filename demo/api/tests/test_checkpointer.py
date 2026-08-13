from unittest.mock import AsyncMock, Mock

import pytest

from lgos_demo_api import checkpointer as demo_checkpointer

POSTGRES_URI = "postgresql://example"


async def test_postgres_runtime_owns_one_ready_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = Mock(name="pool", wait=AsyncMock())
    pool_context = AsyncMock(name="pool_context")
    pool_context.__aenter__.return_value = pool
    saver = Mock(name="saver")
    coordinator = Mock(name="coordinator")
    pool_factory = Mock(return_value=pool_context)
    saver_factory = Mock(return_value=saver)
    coordinator_factory = Mock(return_value=coordinator)
    monkeypatch.setattr(demo_checkpointer, "AsyncConnectionPool", pool_factory)
    monkeypatch.setattr(demo_checkpointer, "AsyncPostgresSaver", saver_factory)
    monkeypatch.setattr(
        demo_checkpointer,
        "PostgresRunCoordinator",
        coordinator_factory,
    )

    async with demo_checkpointer.postgres_runtime(POSTGRES_URI) as runtime:
        assert runtime.checkpointer is saver
        assert runtime.run_coordinator is coordinator
        pool_context.__aenter__.assert_awaited_once_with()
        pool.wait.assert_awaited_once_with()
        pool_context.__aexit__.assert_not_awaited()

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
    coordinator_factory.assert_called_once_with(
        pool,
        max_concurrent_leases=4,
    )
    pool_context.__aexit__.assert_awaited_once_with(None, None, None)


async def test_setup_postgres_schema_runs_saver_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = Mock(setup=AsyncMock())
    saver_context = AsyncMock()
    saver_context.__aenter__.return_value = saver
    saver_factory = Mock(return_value=saver_context)
    monkeypatch.setattr(
        demo_checkpointer.AsyncPostgresSaver,
        "from_conn_string",
        saver_factory,
    )

    await demo_checkpointer.setup_postgres_schema(POSTGRES_URI)

    saver_factory.assert_called_once_with(POSTGRES_URI)
    saver_context.__aenter__.assert_awaited_once_with()
    saver.setup.assert_awaited_once_with()
    saver_context.__aexit__.assert_awaited_once_with(None, None, None)
