"""OpenAI chat protocol helpers for LangGraph interrupts."""

import json
from typing import Any

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequestMessage,
    Role,
)

INTERRUPT_TOOL_NAME = "langgraph_interrupt"
INTERRUPT_TOOL_CALL_ID_PREFIX = "lg_interrupt_"
INTERRUPT_ARGUMENT_VERSION = 1
INTERRUPT_ARGUMENT_KIND = "hitl.interrupt"

NO_RESUME = object()


class InvalidResumeRequestError(ValueError):
    """Raised when a tool response cannot be used to resume an interrupt."""


def interrupt_tool_call_id(interrupt_id: str) -> str:
    return f"{INTERRUPT_TOOL_CALL_ID_PREFIX}{interrupt_id}"


def interrupt_arguments(
    *,
    thread_id: str,
    interrupt_id: str,
    payload: Any,
) -> str:
    # Graph-owned payloads may contain application objects; keep the OpenAI tool
    # call serializable without constraining the graph's internal value types.
    return json.dumps(
        {
            "version": INTERRUPT_ARGUMENT_VERSION,
            "kind": INTERRUPT_ARGUMENT_KIND,
            "thread_id": thread_id,
            "interrupt_id": interrupt_id,
            "payload": payload,
        },
        default=str,
    )


def parse_resume_value(
    messages: list[ChatCompletionRequestMessage],
) -> Any:
    if not messages or messages[-1].role != Role.TOOL:
        return NO_RESUME

    tool_message = messages[-1]
    if not tool_message.tool_call_id:
        raise InvalidResumeRequestError(
            "Interrupt resume tool messages must include tool_call_id."
        )

    if not tool_message.tool_call_id.startswith(INTERRUPT_TOOL_CALL_ID_PREFIX):
        raise InvalidResumeRequestError(
            "Tool response is not for a LangGraph interrupt tool call."
        )

    if tool_message.tool_call_id == INTERRUPT_TOOL_CALL_ID_PREFIX:
        raise InvalidResumeRequestError("Interrupt resume tool_call_id is invalid.")

    try:
        payload = json.loads(tool_message.content or "")
    except json.JSONDecodeError as e:
        raise InvalidResumeRequestError(
            'Interrupt resume tool content must be JSON like {"resume": "..."}'
        ) from e

    if not isinstance(payload, dict) or "resume" not in payload:
        raise InvalidResumeRequestError(
            'Interrupt resume tool content must be JSON like {"resume": "..."}'
        )

    return payload["resume"]
