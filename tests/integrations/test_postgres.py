from asyncio import CancelledError
from unittest.mock import ANY, AsyncMock, Mock, call

import pytest

from langgraph_openai_serve.graph.coordination import RunBusyError
from langgraph_openai_serve.integrations import postgres

THREAD_1_LOCK_KEY = 5407239785987761849
THREAD_NEGATIVE_LOCK_KEY = -7821029440514528571


def _coordinator_for(
    connection: Mock,
) -> tuple[postgres.PostgresRunCoordinator, Mock, AsyncMock]:
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = connection
    pool = Mock(connection=Mock(return_value=connection_context))
    return (
        postgres.PostgresRunCoordinator(pool, max_concurrent_leases=1),
        pool,
        connection_context,
    )


def test_coordinator_rejects_close_returns_pool() -> None:
    pool = Mock(close_returns=True)

    with pytest.raises(ValueError, match="close_returns=False"):
        postgres.PostgresRunCoordinator(pool, max_concurrent_leases=1)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), True])
def test_coordinator_rejects_invalid_capacity(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        postgres.PostgresRunCoordinator(Mock(), max_concurrent_leases=value)


async def test_coordinator_holds_and_releases_session_lock() -> None:
    lock_cursor = Mock(fetchone=AsyncMock(return_value={"acquired": True}))
    unlock_cursor = Mock(fetchone=AsyncMock(return_value={"released": True}))
    connection = Mock(
        execute=AsyncMock(side_effect=[lock_cursor, unlock_cursor]),
        close=AsyncMock(),
    )
    coordinator, pool, connection_context = _coordinator_for(connection)
    lock_key = postgres._advisory_lock_key("thread-1")

    with pytest.raises(RuntimeError, match="run failed"):
        async with coordinator("thread-1"):
            connection_context.__aexit__.assert_not_awaited()
            raise RuntimeError("run failed")

    assert connection.execute.await_args_list == [
        call(postgres._TRY_ADVISORY_LOCK_SQL, (lock_key,)),
        call(postgres._UNLOCK_ADVISORY_LOCK_SQL, (lock_key,)),
    ]
    unlock_cursor.fetchone.assert_awaited_once_with()
    connection.close.assert_not_awaited()
    connection_context.__aexit__.assert_awaited_once_with(
        RuntimeError,
        ANY,
        ANY,
    )
    pool.connection.assert_called_once_with()


async def test_coordinator_rejects_an_occupied_database_lock() -> None:
    lock_cursor = Mock(fetchone=AsyncMock(return_value={"acquired": False}))
    connection = Mock(
        execute=AsyncMock(return_value=lock_cursor),
        close=AsyncMock(),
    )
    coordinator, _, _ = _coordinator_for(connection)

    with pytest.raises(RunBusyError) as exc_info:
        async with coordinator("thread-1"):
            pass

    assert exc_info.value.key == "thread-1"
    connection.execute.assert_awaited_once_with(
        postgres._TRY_ADVISORY_LOCK_SQL,
        (postgres._advisory_lock_key("thread-1"),),
    )
    connection.close.assert_not_awaited()


@pytest.mark.parametrize(
    ("failure_stage", "failure"),
    [
        pytest.param("execute", CancelledError(), id="execute-cancelled"),
        pytest.param("execute", RuntimeError("execute failed"), id="execute-error"),
        pytest.param("fetchone", CancelledError(), id="fetchone-cancelled"),
        pytest.param(
            "fetchone",
            RuntimeError("fetchone failed"),
            id="fetchone-error",
        ),
    ],
)
async def test_coordinator_discards_indeterminate_acquisition(
    failure_stage: str,
    failure: BaseException,
) -> None:
    lock_cursor = Mock(fetchone=AsyncMock())
    if failure_stage == "execute":
        execute = AsyncMock(side_effect=failure)
    else:
        execute = AsyncMock(return_value=lock_cursor)
        lock_cursor.fetchone.side_effect = failure
    connection = Mock(execute=execute, close=AsyncMock())
    coordinator, _, _ = _coordinator_for(connection)

    with pytest.raises(type(failure)) as exc_info:
        async with coordinator("thread-1"):
            pass

    assert exc_info.value is failure
    connection.close.assert_awaited_once_with()


async def test_coordinator_discards_acquisition_without_result() -> None:
    lock_cursor = Mock(fetchone=AsyncMock(return_value=None))
    connection = Mock(
        execute=AsyncMock(return_value=lock_cursor),
        close=AsyncMock(),
    )
    coordinator, _, _ = _coordinator_for(connection)

    with pytest.raises(RuntimeError, match="acquisition returned no result"):
        async with coordinator("thread-1"):
            pass

    connection.close.assert_awaited_once_with()


async def test_coordinator_reserves_pool_capacity_for_checkpoints() -> None:
    lock_cursor = Mock(fetchone=AsyncMock(return_value={"acquired": True}))
    unlock_cursor = Mock(fetchone=AsyncMock(return_value={"released": True}))
    connection = Mock(execute=AsyncMock(side_effect=[lock_cursor, unlock_cursor]))
    coordinator, pool, _ = _coordinator_for(connection)

    async with coordinator("thread-1"):
        with pytest.raises(RunBusyError) as exc_info:
            async with coordinator("thread-2"):
                pass

    assert exc_info.value.key == "thread-2"
    pool.connection.assert_called_once_with()


@pytest.mark.parametrize(
    ("unlock_error", "body_error", "expected_error", "match"),
    [
        pytest.param(
            None,
            None,
            RuntimeError,
            "could not be released",
            id="unlock-failure",
        ),
        pytest.param(
            None,
            ValueError("run failed"),
            ValueError,
            "run failed",
            id="body-error-wins",
        ),
        pytest.param(
            CancelledError(),
            ValueError("run failed"),
            CancelledError,
            None,
            id="unlock-cancellation-wins",
        ),
    ],
)
async def test_coordinator_discards_session_after_unlock_failure(
    unlock_error: BaseException | None,
    body_error: Exception | None,
    expected_error: type[BaseException],
    match: str | None,
) -> None:
    lock_cursor = Mock(fetchone=AsyncMock(return_value={"acquired": True}))
    unlock_cursor = Mock(
        fetchone=AsyncMock(
            return_value={"released": False},
            side_effect=unlock_error,
        )
    )
    connection = Mock(
        execute=AsyncMock(side_effect=[lock_cursor, unlock_cursor]),
        close=AsyncMock(),
    )
    coordinator, _, _ = _coordinator_for(connection)

    with pytest.raises(expected_error, match=match):
        async with coordinator("thread-1"):
            if body_error is not None:
                raise body_error

    connection.close.assert_awaited_once_with()


def test_advisory_lock_key_is_stable_signed_bigint() -> None:
    lock_key = postgres._advisory_lock_key("thread-1")

    assert lock_key == THREAD_1_LOCK_KEY
    assert -(2**63) <= lock_key < 2**63
    assert postgres._advisory_lock_key("thread-negative") == THREAD_NEGATIVE_LOCK_KEY
