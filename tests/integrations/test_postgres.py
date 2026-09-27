from asyncio import CancelledError
from unittest.mock import AsyncMock, Mock

import pytest
from anyio import fail_after

from langgraph_openai_serve.graph.interrupt import RunBusyError
from langgraph_openai_serve.integrations import postgres


def _connection(*rows: dict[str, bool]) -> Mock:
    cursors = [Mock(fetchone=AsyncMock(return_value=row)) for row in rows]
    return Mock(execute=AsyncMock(side_effect=cursors), close=AsyncMock())


def _coordinator_for(connection: Mock) -> postgres.PostgresRunCoordinator:
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = connection
    pool = Mock(close_returns=False, connection=Mock(return_value=connection_context))
    return postgres.PostgresRunCoordinator(pool, max_concurrent_leases=1)


@pytest.mark.parametrize(
    ("pool", "capacity", "match"),
    [
        pytest.param(Mock(close_returns=True), 1, "close_returns=False", id="pool"),
        pytest.param(Mock(close_returns=False), 0, "positive integer", id="capacity"),
    ],
)
def test_coordinator_rejects_unsafe_configuration(pool, capacity, match) -> None:
    with pytest.raises(ValueError, match=match):
        postgres.PostgresRunCoordinator(pool, max_concurrent_leases=capacity)


async def test_coordinator_releases_its_lock_and_keeps_the_session() -> None:
    connection = _connection({"acquired": True}, {"released": True})
    coordinator = _coordinator_for(connection)

    async def fail_run() -> None:
        async with coordinator("thread-1"):
            msg = "run failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="run failed"):
        await fail_run()

    assert "pg_advisory_unlock" in connection.execute.await_args.args[0]
    connection.close.assert_not_awaited()


async def test_coordinator_rejects_an_occupied_database_lock() -> None:
    connection = _connection({"acquired": False})
    coordinator = _coordinator_for(connection)

    with pytest.raises(RunBusyError):
        async with coordinator("thread-1"):
            pass

    connection.close.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    [CancelledError(), RuntimeError("execute failed")],
    ids=["cancelled", "error"],
)
async def test_coordinator_discards_indeterminate_acquisition(
    failure: BaseException,
) -> None:
    connection = Mock(execute=AsyncMock(side_effect=failure), close=AsyncMock())
    coordinator = _coordinator_for(connection)

    with pytest.raises(type(failure)):
        async with coordinator("thread-1"):
            pass

    connection.close.assert_awaited_once_with()


async def test_coordinator_discards_session_after_unlock_failure() -> None:
    connection = _connection({"acquired": True}, {"released": False})
    coordinator = _coordinator_for(connection)

    with pytest.raises(RuntimeError, match="could not be released"):
        async with coordinator("thread-1"):
            pass

    connection.close.assert_awaited_once_with()


async def test_coordinator_reserves_pool_capacity_for_checkpoints() -> None:
    connection = _connection({"acquired": True}, {"released": True})
    coordinator = _coordinator_for(connection)

    async with coordinator("thread-1"):
        with fail_after(1), pytest.raises(RunBusyError):
            async with coordinator("thread-2"):
                pass
