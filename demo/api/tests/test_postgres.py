from unittest.mock import AsyncMock, Mock

import pytest

from lgos_demo_api.persistence import postgres as demo_postgres

POSTGRES_URI = "postgresql://example"


async def test_postgres_runtime_owns_one_ready_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migrate = AsyncMock()
    monkeypatch.setattr(demo_postgres, "setup_postgres_schema", migrate)
    pool = Mock(name="pool", wait=AsyncMock())
    pool_context = AsyncMock(name="pool_context")
    pool_context.__aenter__.return_value = pool
    saver = Mock(name="saver")
    store = Mock(name="store")
    coordinator = Mock(name="coordinator")
    pool_factory = Mock(return_value=pool_context)
    saver_factory = Mock(return_value=saver)
    store_factory = Mock(return_value=store)
    coordinator_factory = Mock(return_value=coordinator)
    monkeypatch.setattr(demo_postgres, "AsyncConnectionPool", pool_factory)
    monkeypatch.setattr(demo_postgres, "AsyncPostgresSaver", saver_factory)
    monkeypatch.setattr(demo_postgres, "AsyncPostgresStore", store_factory)
    monkeypatch.setattr(
        demo_postgres,
        "PostgresRunCoordinator",
        coordinator_factory,
    )

    async with demo_postgres.postgres_runtime(POSTGRES_URI) as runtime:
        migrate.assert_awaited_once_with(POSTGRES_URI)
        assert runtime.checkpointer is saver
        assert runtime.store is store
        assert runtime.run_coordinator is coordinator
        pool_context.__aenter__.assert_awaited_once_with()
        pool.wait.assert_awaited_once_with()
        pool_context.__aexit__.assert_not_awaited()

    pool_factory.assert_called_once_with(
        conninfo=POSTGRES_URI,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": demo_postgres.dict_row,
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
    pool_context.__aexit__.assert_awaited_once_with(None, None, None)


async def test_postgres_runtime_does_not_start_when_migration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        demo_postgres,
        "setup_postgres_schema",
        AsyncMock(side_effect=RuntimeError("migration failed")),
    )

    with pytest.raises(RuntimeError, match="migration failed"):
        async with demo_postgres.postgres_runtime(POSTGRES_URI):
            pytest.fail("The runtime must not serve work after failed migrations")
