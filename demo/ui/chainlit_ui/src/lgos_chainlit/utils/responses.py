"""Responses API helpers for Chainlit models."""

from collections.abc import Mapping, Sequence
from typing import Any

import chainlit as cl
from chainlit_utils.chat import mark_model_context_excluded
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
)
from plotly import io as pio
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from lgos_chainlit.utils.clients import files_request

DISPLAY_FILE_TOOL_NAME = "display_file"
PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"
RESPONSE_OUTPUT = TypeAdapter(list[ResponseOutputItem])


class DisplayFileArguments(BaseModel):
    """Arguments for the client-owned file display function."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    media_type: str = Field(pattern=r"^(?:image/|application/vnd\.plotly\.v1\+json$)")
    title: str = Field(min_length=1)
    alt: str = Field(min_length=1)


DISPLAY_FILE_TOOL: FunctionToolParam = {
    "type": "function",
    "name": DISPLAY_FILE_TOOL_NAME,
    "description": "Display a file stored in the configured OpenAI Files API.",
    "strict": True,
    "parameters": DisplayFileArguments.model_json_schema(),
}


class CommentaryTaskList:
    """Render streamed commentary as one native Chainlit task list."""

    def __init__(self) -> None:
        self._task_list: cl.TaskList | None = None
        self._active_task: cl.Task | None = None

    async def add(self, content: str) -> None:
        """Complete the prior task and append the latest status as running."""
        if not content:
            return
        if self._task_list is None:
            self._task_list = cl.TaskList()
        if self._active_task is not None:
            self._active_task.status = cl.TaskStatus.DONE

        task = cl.Task(title=content, status=cl.TaskStatus.RUNNING)
        await self._task_list.add_task(task)
        self._task_list.status = "Running..."
        self._active_task = task
        await self._task_list.send()

    async def complete(self) -> None:
        """Mark the task list complete after the full Responses loop succeeds."""
        if self._task_list is None:
            return
        if self._active_task is not None:
            self._active_task.status = cl.TaskStatus.DONE
            self._active_task = None
        self._task_list.status = "Done"
        await self._task_list.send()

    async def stop(self) -> None:
        """Mark the active task as failed when the Responses loop stops early."""
        if self._task_list is None or self._active_task is None:
            return
        self._active_task.status = cl.TaskStatus.FAILED
        self._active_task = None
        self._task_list.status = "Stopped"
        await self._task_list.send()


def response_input(messages: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    """Convert Chainlit's text transcript to Responses message items."""
    items = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            continue
        item = {"role": role, "content": content}
        if role == "assistant":
            item["phase"] = message.get("phase") or "final_answer"
        items.append(item)
    return items


def final_answer(response: Response) -> str:
    """Return durable final-answer text without concatenating commentary."""
    parts = []
    for item in response.output:
        if item.type != "message" or item.phase == "commentary":
            continue
        parts.extend(part.text for part in item.content if part.type == "output_text")
    return "".join(parts)


def raise_for_response(response: Response) -> None:
    """Only completed Responses may be rendered or continued as successful."""
    if response.status == "completed":
        return
    detail = response.error
    raise RuntimeError(detail.message if detail is not None else "Response failed.")


def function_calls(response: Response) -> list[ResponseFunctionToolCall]:
    """Return client-owned function calls from a completed Response."""
    return [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]


async def display_file(call: ResponseFunctionToolCall) -> dict[str, str]:
    """Download and persist a native image or interactive Plotly element."""
    if call.name != DISPLAY_FILE_TOOL_NAME:
        msg = f"Unsupported client function: {call.name}"
        raise ValueError(msg)
    try:
        arguments = DisplayFileArguments.model_validate_json(call.arguments)
    except ValueError as exc:
        msg = "The display_file call contains invalid arguments."
        raise ValueError(msg) from exc

    client, provider = files_request()
    download = await client.files.content(
        arguments.file_id, extra_query={"provider": provider}
    )
    content = await download.aread()
    if arguments.media_type == PLOTLY_MEDIA_TYPE:
        element = cl.Plotly(
            name=arguments.filename,
            figure=pio.from_json(content.decode()),
            display="inline",
        )
    else:
        element = cl.Image(
            name=arguments.filename,
            content=content,
            mime=arguments.media_type,
            display="inline",
        )
    message = cl.Message(content=arguments.title, elements=[element])
    mark_model_context_excluded(message)
    await message.send()
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"displayed":true}',
    }


def continuation_input(
    response: Response,
    outputs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Replay complete output items followed by matching function results."""
    return [
        # Serialize wire types, excluding SDK-only parsed fields on subclasses.
        *RESPONSE_OUTPUT.dump_python(response.output, mode="json", exclude_none=True),
        *outputs,
    ]
