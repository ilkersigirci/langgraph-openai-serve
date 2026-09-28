from collections.abc import Iterator
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from lgos_chainlit.settings import get_chainlit_settings


@pytest.fixture
def application(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "test-signing-secret")
    get_chainlit_settings.cache_clear()
    # Authentication tests already exercise the mounted Chainlit singleton.
    # This fixture owns only the host app's startup/shutdown lifecycle.
    monkeypatch.setattr("chainlit.utils.mount_chainlit", Mock())
    from lgos_chainlit import main

    monkeypatch.setattr(main.gateway_http_client, "aclose", AsyncMock())
    monkeypatch.setattr(main, "_close_chainlit_data_layer", AsyncMock())
    yield main
    get_chainlit_settings.cache_clear()


async def test_every_startup_migrates_before_serving(
    application: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrate = AsyncMock()
    monkeypatch.setattr(application, "setup_chainlit_schema", migrate)

    for startup in range(1, 3):
        async with application.app.router.lifespan_context(application.app):
            assert migrate.await_count == startup
            migrate.assert_awaited_with("postgresql://test:test@localhost/test")


async def test_failed_migration_prevents_startup_and_closes_clients(
    application: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        application,
        "setup_chainlit_schema",
        AsyncMock(side_effect=RuntimeError("migration failed")),
    )

    with pytest.raises(RuntimeError, match="migration failed"):
        async with application.app.router.lifespan_context(application.app):
            pytest.fail("Chainlit must not serve requests after failed migrations")

    application.gateway_http_client.aclose.assert_awaited_once()
    application._close_chainlit_data_layer.assert_awaited_once()
