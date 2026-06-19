"""Helpers for response-required UI events carried as OpenAI tool calls."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequestMessage,
    Role,
    ToolCall,
    ToolCallFunction,
)

UI_EVENT_TOOL_NAME = "ui_event"


class UIEventInterrupt(Exception):
    """Raised when a LangGraph interrupt should become a UI-event tool call."""

    def __init__(self, interrupt: Any) -> None:
        self.interrupt = interrupt
        super().__init__("LangGraph interrupted for a UI-event response")


def make_ui_event_tool_call(event: dict[str, Any]) -> ToolCall:
    """Wrap a UI-event envelope in the reserved OpenAI tool call."""
    return ToolCall(
        id=f"call-{uuid.uuid4()}",
        function=ToolCallFunction(
            name=UI_EVENT_TOOL_NAME,
            arguments=json.dumps(event, separators=(",", ":"), ensure_ascii=False),
        ),
    )


def parse_ui_event_tool_response(
    message: ChatCompletionRequestMessage,
    *,
    expected_tool_call_id: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate lightweight correlation for a UI-event tool response."""
    if message.role != Role.TOOL:
        raise ValueError("UI-event response must use role='tool'")
    if expected_tool_call_id is not None and message.name not in (
        None,
        UI_EVENT_TOOL_NAME,
    ):
        raise ValueError("UI-event tool response has an unexpected tool name")

    tool_call_id = getattr(message, "tool_call_id", None)
    if expected_tool_call_id is not None and tool_call_id != expected_tool_call_id:
        raise ValueError("UI-event tool response does not match the tool call id")

    try:
        value = json.loads(message.content or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("UI-event tool response content must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("UI-event tool response content must be a JSON object")
    if expected_run_id is not None and value.get("runId") != expected_run_id:
        raise ValueError("UI-event tool response does not match the run id")
    return value


def extract_interrupt(result: Any) -> Any | None:
    """Return the first LangGraph interrupt object from a graph result."""
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0]
