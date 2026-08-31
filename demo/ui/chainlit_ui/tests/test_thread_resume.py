import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def thread_resume() -> Any:
    return importlib.import_module("lgos_chainlit.utils.thread_resume")


async def test_schedule_after_thread_hydration_tracks_the_chainlit_task(
    monkeypatch: pytest.MonkeyPatch,
    thread_resume: Any,
) -> None:
    callback = AsyncMock()
    session = SimpleNamespace(current_task=None)
    monkeypatch.setattr(thread_resume.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        thread_resume,
        "chainlit_context",
        SimpleNamespace(session=session),
    )

    task = thread_resume.schedule_after_thread_hydration(callback)
    await task

    assert session.current_task is task
    callback.assert_awaited_once_with()


def test_reuse_persisted_step_copies_chainlit_identity(thread_resume: Any) -> None:
    source = Mock(
        id="persisted-message",
        parent_id="parent-message",
        created_at="2026-08-10T12:00:00Z",
        metadata={"state": "pending"},
    )
    target = SimpleNamespace(
        id="transient-message",
        parent_id=None,
        created_at=None,
        metadata=None,
        persisted=False,
    )

    thread_resume.reuse_persisted_step(target, source)

    assert target.id == source.id
    assert target.parent_id == source.parent_id
    assert target.created_at == source.created_at
    assert target.metadata is source.metadata
    assert target.persisted is True
