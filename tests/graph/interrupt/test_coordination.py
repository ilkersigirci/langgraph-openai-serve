import pytest
from anyio import fail_after

from langgraph_openai_serve.graph.interrupt import (
    InMemoryRunCoordinator,
    RunBusyError,
)


async def test_in_memory_coordinator_rejects_an_occupied_key_without_waiting() -> None:
    coordinator = InMemoryRunCoordinator()

    async with coordinator("thread-1"):
        with fail_after(1), pytest.raises(RunBusyError) as exc_info:
            async with coordinator("thread-1"):
                pass

    assert exc_info.value.key == "thread-1"


async def test_in_memory_coordinator_allows_distinct_keys() -> None:
    coordinator = InMemoryRunCoordinator()

    async with coordinator("thread-1"), coordinator("thread-2"):
        pass


async def test_in_memory_coordinator_releases_after_failure() -> None:
    coordinator = InMemoryRunCoordinator()

    msg = "run failed"
    with pytest.raises(RuntimeError, match=msg):
        async with coordinator("thread-1"):
            raise RuntimeError(msg)

    async with coordinator("thread-1"):
        pass
