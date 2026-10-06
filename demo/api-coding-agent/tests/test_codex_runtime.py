import asyncio
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph_openai_serve import GraphError
from openai_codex import ApprovalMode, AsyncCodex, Sandbox
from openai_codex.models import Notification, UnknownNotification

from lgos_api_coding_agent.codex_runtime import CodexTurn, own_codex, runtime_events
from lgos_api_coding_agent.settings import RuntimeSettings


def runtime_settings(workspace: Path, timeout_seconds: float = 10) -> RuntimeSettings:
    return RuntimeSettings(
        model="fixture",
        base_url="http://model/v1",
        api_key="test",
        workspace=workspace,
        timeout_seconds=timeout_seconds,
    )


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
        msg = "Runtime closed during initialize"
        raise RuntimeError(msg)

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

    source = runtime_events(runtime_settings(tmp_path), factory=WorkspaceCodex)
    first = source(CodexTurn("first", "first", None, continues=False))
    second = source(CodexTurn("second", "second", None, continues=False))
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


async def test_named_conversation_continues_its_codex_thread(tmp_path) -> None:
    names: dict[str, str] = {}
    prompts: list[tuple[str, str]] = []

    class Thread:
        def __init__(self, identifier: str) -> None:
            self.id = identifier

        async def set_name(self, name: str) -> None:
            names[self.id] = name

        async def turn(self, prompt: str):
            prompts.append((self.id, prompt))
            return self

        async def stream(self):
            yield Notification("turn/completed", UnknownNotification({}))

    class ConversationCodex(AsyncCodex):
        """Each request gets a new runtime; threads outlive it as in CODEX_HOME."""

        def __init__(self, config) -> None:
            pass

        async def __aenter__(self):
            return self

        async def close(self) -> None:
            pass

        async def thread_list(self, *, search_term: str):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(id=identifier, name=name)
                    for identifier, name in names.items()
                    if search_term in name
                ]
            )

        async def thread_start(self, **options):
            assert options["ephemeral"] is False
            return Thread(f"thread-{len(names)}")

        async def thread_resume(self, thread_id: str, **options):
            return Thread(thread_id)

    source = runtime_events(runtime_settings(tmp_path), factory=ConversationCodex)
    turns = [
        CodexTurn("whole history", "first message", "chat", continues=False),
        CodexTurn("longer history", "second message", "chat", continues=True),
        CodexTurn("other history", "other message", "other chat", continues=True),
    ]
    async with asyncio.timeout(3):
        streams = [[item async for item in source(turn)] for turn in turns]

    assert prompts == [
        ("thread-0", "whole history"),
        ("thread-0", "second message"),
        ("thread-1", "other history"),
    ]
    # Only the conversation whose earlier turns Codex does not hold is told so.
    assert [isinstance(stream[0], str) for stream in streams] == [False, False, True]


async def test_time_limit_fails_the_turn_while_its_consumer_is_busy(tmp_path) -> None:
    class IdleCodex(AsyncCodex):
        def __init__(self, config) -> None:
            pass

        async def __aenter__(self):
            return self

        async def close(self) -> None:
            pass

        async def thread_start(self, **options):
            return self

        async def turn(self, prompt):
            return self

        async def stream(self):
            yield Notification("ready", UnknownNotification({}))
            await asyncio.Event().wait()

    source = runtime_events(runtime_settings(tmp_path, 0.05), factory=IdleCodex)
    events = source(CodexTurn("prompt", "prompt", None, continues=False))

    async def consume() -> None:
        async with aclosing(events):
            await anext(events)
            # LangChain awaits its token callbacks between events.
            await asyncio.sleep(0.1)
            await anext(events)

    task = asyncio.create_task(consume())
    await asyncio.wait({task}, timeout=3)

    assert task.done()
    assert not task.cancelled()
    with pytest.raises(GraphError, match="time limit"):
        task.result()
