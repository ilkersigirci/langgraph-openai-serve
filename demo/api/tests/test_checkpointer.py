from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from lgos_demo_api import checkpointer as demo_checkpointer

POSTGRES_URI = "postgresql://example"


@pytest.mark.anyio
async def test_postgres_checkpointer_owns_pool_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = Mock(name="pool")
    pool_context = AsyncMock(name="pool_context")
    pool_context.__aenter__.return_value = pool
    saver = Mock(name="saver")
    pool_factory = Mock(return_value=pool_context)
    saver_factory = Mock(return_value=saver)
    monkeypatch.setattr(demo_checkpointer, "AsyncConnectionPool", pool_factory)
    monkeypatch.setattr(demo_checkpointer, "AsyncPostgresSaver", saver_factory)

    async with demo_checkpointer.postgres_checkpointer(POSTGRES_URI) as result:
        assert result is saver
        pool_context.__aenter__.assert_awaited_once_with()
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
    pool_context.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.anyio
async def test_setup_postgres_schema_runs_saver_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = Mock(setup=AsyncMock())

    @asynccontextmanager
    async def checkpointer(postgres_uri: str):
        assert postgres_uri == POSTGRES_URI
        yield saver

    checkpointer_factory = Mock(wraps=checkpointer)
    monkeypatch.setattr(
        demo_checkpointer,
        "postgres_checkpointer",
        checkpointer_factory,
    )

    await demo_checkpointer.setup_postgres_schema(POSTGRES_URI)

    checkpointer_factory.assert_called_once_with(POSTGRES_URI)
    saver.setup.assert_awaited_once_with()
