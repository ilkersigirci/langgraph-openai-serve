import asyncio

import pytest
from openai_codex import AsyncCodex

from lgos_api_coding_agent.codex_runtime import own_codex


class StartingCodex(AsyncCodex):
    """Model the SDK's offloaded start racing a cancelled context entry."""

    def __init__(self, late_start: bool) -> None:
        self.starting = asyncio.Event()
        self.release = asyncio.Event()
        self.late_start = late_start
        self.alive = False
        self.closed = asyncio.Event()

    async def __aenter__(self):
        self.starting.set()
        if self.late_start:
            await self.release.wait()
        self.alive = True
        await self.closed.wait()
        raise RuntimeError("Runtime closed during initialize")

    async def close(self) -> None:
        self.release.set()
        if self.alive:
            self.alive = False
            self.closed.set()


@pytest.mark.parametrize(
    "late_start", [False, True], ids=["during-initialize", "before-process-start"]
)
async def test_cancellation_owns_startup_until_the_runtime_is_closed(
    late_start: bool,
) -> None:
    codex = StartingCodex(late_start)

    async def request() -> None:
        async with own_codex(codex):
            pytest.fail("Cancelled startup must not execute the turn")

    task = asyncio.create_task(request())
    async with asyncio.timeout(3):
        await codex.starting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await codex.closed.wait()
    assert not codex.alive


@pytest.mark.parametrize("shutdown", ["normal", "cancelled"])
async def test_workspace_requests_wait_for_runtime_cleanup(tmp_path, shutdown) -> None:
    from openai_codex import ApprovalMode, Sandbox
    from openai_codex.models import Notification, UnknownNotification

    from lgos_api_coding_agent.codex_runtime import runtime_events
    from lgos_api_coding_agent.settings import RuntimeSettings

    started = asyncio.Queue()
    release_cleanup = asyncio.Event()
    cleanup_started = asyncio.Event()
    launches = []

    class WorkspaceCodex(AsyncCodex):
        async def __aenter__(self):
            return self

        async def thread_start(self, **options):
            # These SDK permissions are the coding service's execution contract.
            assert options["cwd"] == str(tmp_path)
            assert options["sandbox"] is Sandbox.full_access
            assert options["approval_mode"] is ApprovalMode.deny_all
            assert options["ephemeral"] is True
            launches.append(self)
            started.put_nowait(self)
            return self

        async def turn(self, prompt):
            return self

        async def stream(self):
            yield Notification("ready", UnknownNotification({}))
            await asyncio.Event().wait()

        async def close(self):
            cleanup_started.set()
            await release_cleanup.wait()

    settings = RuntimeSettings(
        model="fixture",
        base_url="http://model/v1",
        api_key="test",
        workspace=tmp_path,
        timeout_seconds=10,
    )
    source = runtime_events(settings, factory=WorkspaceCodex)
    first, second = source("first"), source("second")
    second_waiting = asyncio.Event()

    async def start_second():
        second_waiting.set()
        return await anext(second)

    async with asyncio.timeout(3):
        await anext(first)
        await started.get()
        waiting = asyncio.create_task(start_second())
        await second_waiting.wait()
        closing = asyncio.create_task(first.aclose())
        await cleanup_started.wait()
        if shutdown == "cancelled":
            closing.cancel()
        try:
            # Cancelling the closing request must not release the workspace.
            done, _ = await asyncio.wait({closing, waiting}, timeout=0.05)
            assert not done
            assert len(launches) == 1
        finally:
            release_cleanup.set()
            if shutdown == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await closing
            else:
                await closing
        await waiting
        assert len(launches) == 2
        await second.aclose()
