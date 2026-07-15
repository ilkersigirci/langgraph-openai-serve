"""OpenAI message history restoration for persisted Chainlit threads."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import chainlit as cl
from chainlit.step import StepDict
from chainlit.types import ThreadDict

MESSAGE_ROLES = {
    "assistant_message": "assistant",
    "system_message": "system",
    "user_message": "user",
}


def _step_order(item: tuple[int, StepDict]) -> tuple[bool, float, int]:
    index, step = item
    raw_timestamp = step.get("createdAt") or step.get("start") or step.get("end")
    timestamp = None
    if raw_timestamp:
        try:
            parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            timestamp = parsed.timestamp()
        except ValueError:
            pass
    return timestamp is None, timestamp or 0.0, index


def _message_content(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def messages_from_thread(thread: ThreadDict) -> list[dict[str, Any]]:
    """Rebuild portable OpenAI messages from persisted Chainlit message steps."""
    messages: list[dict[str, Any]] = []
    ordered_steps = sorted(enumerate(thread.get("steps", [])), key=_step_order)
    for _, step in ordered_steps:
        role = MESSAGE_ROLES.get(step.get("type", ""))
        output = step.get("output")
        if role is None or output is None or step.get("isError"):
            continue
        messages.append({"role": role, "content": _message_content(output)})
    return messages


def restore_chat_messages(thread: ThreadDict) -> list[dict[str, Any]]:
    """Use Chainlit's restored JSON session, with thread steps as a fallback."""
    restored = cl.user_session.get("messages")
    if isinstance(restored, list) and all(
        isinstance(message, dict) for message in restored
    ):
        return restored

    messages = messages_from_thread(thread)
    cl.user_session.set("messages", messages)
    return messages
