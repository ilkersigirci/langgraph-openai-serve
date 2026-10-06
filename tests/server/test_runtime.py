from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr, ValidationError

from langgraph_openai_serve.server import ServerSettings, runtime
from tests.server.support import server_settings

POSTGRES_URI = "postgresql://example"


async def test_postgres_pool_meets_saver_requirements_and_keeps_a_spare_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_factory = Mock(return_value=AsyncMock())
    coordinator_factory = Mock()
    monkeypatch.setattr(runtime, "_setup_schema", AsyncMock())
    monkeypatch.setattr(runtime, "AsyncConnectionPool", pool_factory)
    monkeypatch.setattr(runtime, "AsyncPostgresSaver", Mock())
    monkeypatch.setattr(runtime, "AsyncPostgresStore", Mock())
    monkeypatch.setattr(runtime, "PostgresRunCoordinator", coordinator_factory)
    settings = server_settings(
        POSTGRES_URI=SecretStr(POSTGRES_URI), POSTGRES_POOL_SIZE=8
    )

    async with runtime.open_resources(settings):
        pass

    # LangGraph's PostgreSQL savers require these settings on a shared pool.
    assert pool_factory.call_args.kwargs["kwargs"] == {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": runtime.dict_row,
    }
    assert pool_factory.call_args.kwargs["max_size"] == settings.POSTGRES_POOL_SIZE
    # One connection stays free for checkpoints while leases hold the others.
    assert coordinator_factory.call_args.kwargs["max_concurrent_leases"] == (
        settings.POSTGRES_POOL_SIZE - 1
    )


async def test_postgres_resources_do_not_open_when_migration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "_setup_schema", AsyncMock(side_effect=RuntimeError("failed"))
    )
    settings = server_settings(POSTGRES_URI=SecretStr(POSTGRES_URI))

    with pytest.raises(RuntimeError, match="failed"):
        async with runtime.open_resources(settings):
            pytest.fail("The server must not serve work after failed migrations")


def test_empty_postgres_uri_keeps_persistence_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `.env` files often leave unused values empty.
    monkeypatch.setenv("LGOS_POSTGRES_URI", "")

    assert ServerSettings().POSTGRES_URI is None


def test_worker_slots_fit_the_postgres_run_leases() -> None:
    uri = SecretStr(POSTGRES_URI)
    slots = 8

    # In-process coordination has no lease limit; each PostgreSQL run needs one.
    server_settings(HATCHET_WORKER_SLOTS=slots)
    server_settings(
        POSTGRES_URI=uri, POSTGRES_POOL_SIZE=slots + 1, HATCHET_WORKER_SLOTS=slots
    )
    with pytest.raises(ValidationError, match="LGOS_POSTGRES_POOL_SIZE"):
        server_settings(POSTGRES_URI=uri, HATCHET_WORKER_SLOTS=slots)
