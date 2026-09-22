import pytest
from anyio import CancelScope, Event, create_task_group, fail_after, sleep_forever

from langgraph_openai_serve.graph.coordination import (
    RunBusyError,
    RunLease,
)


async def test_coordinator_rejects_an_occupied_key_without_waiting(
    coordinator_pair,
) -> None:
    coordinator, peer = coordinator_pair

    async with coordinator("thread-1"):
        with fail_after(1), pytest.raises(RunBusyError) as exc_info:
            async with peer("thread-1"):
                pass

    assert exc_info.value.key == "thread-1"


async def test_coordinator_allows_distinct_keys(coordinator_pair) -> None:
    coordinator, peer = coordinator_pair

    async with coordinator("thread-1"), peer("thread-2"):
        pass


async def test_coordinator_releases_after_failure(coordinator_pair) -> None:
    coordinator, peer = coordinator_pair

    msg = "run failed"
    with pytest.raises(RuntimeError, match=msg):
        async with coordinator("thread-1"):
            raise RuntimeError(msg)

    async with peer("thread-1") as lease:
        assert isinstance(lease, RunLease)
        lease.ensure_owned()


async def test_coordinator_releases_after_cancellation(coordinator_pair):
    coordinator, peer = coordinator_pair
    acquired = Event()
    released = Event()
    scope = CancelScope()
    leases = []

    async def own():
        with scope:
            async with coordinator("thread-1") as lease:
                leases.append(lease)
                acquired.set()
                await sleep_forever()
        released.set()

    with fail_after(5):
        async with create_task_group() as tasks:
            tasks.start_soon(own)
            await acquired.wait()
            scope.cancel()
            await released.wait()
            async with peer("thread-1") as lease:
                assert isinstance(lease, RunLease)
                assert lease is not leases[0]
                lease.ensure_owned()
