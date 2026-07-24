"""Render LGOS client events in Chainlit."""

import chainlit as cl
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from lgos_chainlit.lgos_protocol import (
    LGOS_EXTENSION_KEY,
    ClientEventExtension,
    StatusUpdate,
)
from lgos_chainlit.utils.chat import mark_model_context_excluded


def client_event(chunk: ChatCompletionChunk) -> dict[str, object] | None:
    """Return a validated client event from a completion chunk."""
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict):
        return None

    try:
        parsed = ClientEventExtension.model_validate(extension)
    except ValidationError:
        return None
    return parsed.event.model_dump(mode="json")


def status_update(event: dict[str, object]) -> StatusUpdate | None:
    """Return a portable status update."""
    if event.get("type") != "status":
        return None
    try:
        return StatusUpdate.model_validate(event.get("data"))
    except ValidationError:
        return None


class ClientEventRenderer:
    """Render statuses as native tasks and other events as a timeline."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._element: cl.CustomElement | None = None
        self._task_list: cl.TaskList | None = None
        self._active_task: cl.Task | None = None

    async def render(self, chunk: ChatCompletionChunk) -> None:
        event = client_event(chunk)
        if event is None:
            return

        status = status_update(event)
        if status is not None:
            await self._render_status(status)
            return

        self._events.append(event)
        props = {"events": [*self._events]}
        if self._element is None:
            self._element = cl.CustomElement(
                name="ClientEventTimeline",
                props=props,
            )
            message = cl.Message(content="", elements=[self._element])
            mark_model_context_excluded(message)
            await message.send()
            return

        self._element.props = props
        await self._element.update()

    async def close(self) -> None:
        """Finish a status left open by a cancelled or failed stream."""
        if self._task_list is None or self._active_task is None:
            return

        self._active_task.status = cl.TaskStatus.FAILED
        self._active_task = None
        self._task_list.status = "Stopped"
        await self._task_list.send()

    async def _render_status(self, status: StatusUpdate) -> None:
        if status.hidden:
            if self._task_list is not None:
                await self._task_list.remove()
            self._task_list = None
            self._active_task = None
            return

        if self._task_list is None:
            self._task_list = cl.TaskList()

        if self._active_task is not None:
            self._active_task.status = cl.TaskStatus.DONE
            self._active_task = None

        task = cl.Task(
            title=status.description,
            status=cl.TaskStatus.DONE if status.done else cl.TaskStatus.RUNNING,
        )
        await self._task_list.add_task(task)
        self._task_list.status = "Done" if status.done else "Running..."
        self._active_task = None if status.done else task
        await self._task_list.send()
