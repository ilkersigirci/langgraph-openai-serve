"""Small Chainlit adapters for restoring live UI on a persisted thread."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

import chainlit as cl
from chainlit.context import context as chainlit_context

THREAD_HYDRATION_DELAY_SECONDS = 0.1

_resume_tasks: set[asyncio.Task[None]] = set()


class _ReusableStep(Protocol):
    id: str
    parent_id: str | None
    created_at: str | None
    metadata: dict[str, object] | None
    persisted: bool


def schedule_after_thread_hydration(
    callback: Callable[[], Awaitable[None]],
) -> asyncio.Task[None]:
    """Run UI restoration after Chainlit replaces the client thread state."""
    task = asyncio.create_task(_after_thread_hydration(callback))
    chainlit_context.session.current_task = task
    _resume_tasks.add(task)
    task.add_done_callback(_resume_tasks.discard)
    return task


async def _after_thread_hydration(
    callback: Callable[[], Awaitable[None]],
) -> None:
    # Chainlit has no client-hydrated callback and sends resume_thread only after
    # on_chat_resume returns, so live UI must be emitted on the next short turn.
    await asyncio.sleep(THREAD_HYDRATION_DELAY_SECONDS)
    await callback()


def reuse_persisted_step(target: _ReusableStep, source: cl.Message) -> None:
    """Make a transient Chainlit message update an existing persisted step."""
    target.id = source.id
    target.parent_id = source.parent_id
    target.created_at = source.created_at
    target.metadata = source.metadata
    target.persisted = True
