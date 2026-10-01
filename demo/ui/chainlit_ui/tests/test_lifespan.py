from types import ModuleType
from unittest.mock import AsyncMock

import pytest


async def test_startup_migrates_before_serving(
    application: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrate = AsyncMock()
    monkeypatch.setattr(application, "setup_chainlit_schema", migrate)

    async with application.app.router.lifespan_context(application.app):
        migrate.assert_awaited_once_with("postgresql://test:test@localhost/test")


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
