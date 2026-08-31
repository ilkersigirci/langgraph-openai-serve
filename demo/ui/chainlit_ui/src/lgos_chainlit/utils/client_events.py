"""Render LGOS client events in Chainlit."""

import chainlit as cl
from chainlit_utils.chat import mark_model_context_excluded
from openai.types.chat import ChatCompletionChunk
from plotly import graph_objects as go
from pydantic import ValidationError

from lgos_chainlit.lgos_protocol import (
    LGOS_EXTENSION_KEY,
    ChartArtifact,
    ClientEventExtension,
    StatusUpdate,
)


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


def chart_artifact(event: dict[str, object]) -> ChartArtifact | None:
    """Return a supported chart artifact."""
    if event.get("type") != "artifact":
        return None
    try:
        return ChartArtifact.model_validate(event.get("data"))
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

        artifact = chart_artifact(event)
        if artifact is not None:
            await self._render_chart(artifact)
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

    @staticmethod
    async def _render_chart(artifact: ChartArtifact) -> None:
        trace_type = "bar" if artifact.chart_type == "bar" else "scatter"
        data: list[dict[str, object]] = []
        for series in artifact.series:
            trace: dict[str, object] = {
                "type": trace_type,
                "name": series.name,
                "x": artifact.labels,
                "y": series.values,
                "showlegend": artifact.show_legend,
            }
            if artifact.chart_type == "line":
                trace["mode"] = "lines+markers"
            data.append(trace)
        element = cl.Plotly(
            name=artifact.id,
            figure=go.Figure(
                data=data,
                layout={
                    "title": {"text": artifact.title},
                    "xaxis": {"title": {"text": artifact.x_axis_title}},
                    "yaxis": {"title": {"text": artifact.y_axis_title}},
                    "showlegend": artifact.show_legend,
                },
            ),
            display="inline",
        )
        message = cl.Message(content=artifact.title, elements=[element])
        mark_model_context_excluded(message)
        await message.send()
