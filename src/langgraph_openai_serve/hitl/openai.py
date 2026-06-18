"""OpenAI-compatible representation of LangGraph human interrupts."""

import json
import uuid
from dataclasses import dataclass
from typing import Any

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage

HITL_TOOL_NAME = "langgraph_interrupt"
_HITL_CALL_PREFIX = "call_hitl_"


@dataclass(frozen=True)
class HitlResume:
    """A LangGraph resume command decoded from an OpenAI tool response."""

    thread_id: str
    value: Any


@dataclass(frozen=True)
class HitlToolCall:
    """An OpenAI tool call representing a paused LangGraph execution."""

    id: str
    arguments: str


def parse_resume_message(
    messages: list[ChatCompletionRequestMessage],
) -> HitlResume | None:
    """Return resume data when the request ends with a HITL tool response."""
    if not messages:
        return None

    message = messages[-1]
    if message.role != "tool" or not message.tool_call_id:
        return None
    if not message.tool_call_id.startswith(_HITL_CALL_PREFIX):
        return None

    try:
        value = json.loads(message.content or "null")
    except json.JSONDecodeError:
        value = message.content

    token = message.tool_call_id.removeprefix(_HITL_CALL_PREFIX)
    thread_id = token.split("_", maxsplit=1)[0]
    return HitlResume(thread_id=thread_id, value=value)


def create_tool_call(thread_id: str, interrupts: Any) -> HitlToolCall:
    """Encode LangGraph interrupts as one OpenAI function tool call."""
    tool_call_id = f"{_HITL_CALL_PREFIX}{thread_id}_{uuid.uuid4().hex}"
    arguments = json.dumps(
        {
            "interrupts": [
                {"id": str(item.id), "value": item.value} for item in interrupts
            ]
        }
    )
    return HitlToolCall(id=tool_call_id, arguments=arguments)


def tool_call_message_payload(tool_call: HitlToolCall) -> dict[str, Any]:
    """Return the non-streaming assistant tool call payload."""
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": HITL_TOOL_NAME,
            "arguments": tool_call.arguments,
        },
    }


def tool_call_delta_payload(tool_call: HitlToolCall, index: int = 0) -> dict[str, Any]:
    """Return the streaming assistant tool call delta payload."""
    return {
        "index": index,
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": HITL_TOOL_NAME,
            "arguments": tool_call.arguments,
        },
    }
