from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr

from langgraph_openai_serve.server import (
    ServerSettings,
    hatchet as server_hatchet,
    runtime,
)
from tests.server.support import create_registry, server_settings

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
    # Smaller than the default worker slots: only `lgos worker` uses those.
    settings = server_settings(
        POSTGRES_URI=SecretStr(POSTGRES_URI), POSTGRES_POOL_SIZE=2
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"BACKGROUND": "memory"}, "LGOS_BACKGROUND=hatchet"),
        ({"HATCHET_WORKER_SLOTS": 5}, "LGOS_POSTGRES_POOL_SIZE"),
        ({"HATCHET_WORKER_SLOTS": 8}, "LGOS_POSTGRES_POOL_SIZE"),
    ],
    ids=["not-hatchet", "slots-equal-pool", "slots-above-pool"],
)
def test_worker_rejects_settings_it_cannot_run(
    overrides: dict[str, object], message: str
) -> None:
    settings = server_settings(
        **{
            "BACKGROUND": "hatchet",
            "POSTGRES_URI": SecretStr(POSTGRES_URI),
            "POSTGRES_POOL_SIZE": 5,
            **overrides,
        }
    )

    with pytest.raises(ValueError, match=message):
        server_hatchet.run_worker(create_registry, settings)


@pytest.mark.parametrize(
    ("uri", "pool_size", "slots"),
    [(None, 2, 8), (SecretStr(POSTGRES_URI), 5, 4)],
    ids=["memory-has-no-lease-limit", "postgres-keeps-a-spare-connection"],
)
def test_worker_starts_when_slots_fit_available_resources(
    monkeypatch: pytest.MonkeyPatch, uri: SecretStr | None, pool_size: int, slots: int
) -> None:
    hatchet = Mock()
    monkeypatch.setattr(server_hatchet, "create_hatchet", lambda: hatchet)
    settings = server_settings(
        BACKGROUND="hatchet",
        POSTGRES_URI=uri,
        POSTGRES_POOL_SIZE=pool_size,
        HATCHET_WORKER_SLOTS=slots,
    )

    server_hatchet.run_worker(create_registry, settings)

    assert hatchet.worker.call_args.kwargs["slots"] == slots
    hatchet.worker.return_value.start.assert_called_once_with()
