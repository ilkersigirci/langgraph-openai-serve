"""Adapt Responses interrupt calls to Open WebUI's native ask-user UI."""

import base64
import json
from typing import Any, cast

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, Field, field_validator

from .contracts import (
    ASK_USER_CALL_ID_PREFIX,
    ASK_USER_MAX_QUESTIONS,
    ASK_USER_QUESTION_MAX_LENGTH,
    ASK_USER_REJECTED_OUTPUT,
    ASK_USER_TOOL_NAME,
    INTERRUPT_TOOL_NAME,
    InterruptCancelled,
)


class InterruptCursor(BaseModel):
    """Responses continuation persisted by Open WebUI with its ask-user call."""

    previous_response_id: str = Field(min_length=1)
    calls: list[ResponseFunctionToolCall] = Field(
        min_length=1, max_length=ASK_USER_MAX_QUESTIONS
    )

    @field_validator("calls")
    @classmethod
    def validate_calls(
        cls, calls: list[ResponseFunctionToolCall]
    ) -> list[ResponseFunctionToolCall]:
        if any(call.name != INTERRUPT_TOOL_NAME or not call.call_id for call in calls):
            msg = "LangGraph API returned an invalid interrupt batch."
            raise ValueError(msg)
        if len({call.call_id for call in calls}) != len(calls):
            msg = "LangGraph API returned duplicate interrupt call IDs."
            raise ValueError(msg)
        return calls


def _ask_user_to_resume(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], str] | None:
    """Restore a Responses continuation from Open WebUI's persisted answer."""
    if not messages:
        return None

    assistant_index = len(messages) - 1
    while (
        assistant_index >= 0
        and isinstance(messages[assistant_index], dict)
        and messages[assistant_index].get("role") == "tool"
    ):
        assistant_index -= 1
    if assistant_index < 0:
        return None
    assistant = messages[assistant_index]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return None

    tool_calls = assistant.get("tool_calls")
    if not isinstance(tool_calls, list) or not any(
        isinstance(call, dict)
        and isinstance(call.get("id"), str)
        and call["id"].startswith(ASK_USER_CALL_ID_PREFIX)
        for call in tool_calls
    ):
        return None
    if len(tool_calls) != 1 or assistant_index != len(messages) - 2:
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)
    ask_call = tool_calls[0]
    function = ask_call.get("function") if isinstance(ask_call, dict) else None
    call_id = ask_call.get("id") if isinstance(ask_call, dict) else None
    if (
        not isinstance(function, dict)
        or function.get("name") != ASK_USER_TOOL_NAME
        or not isinstance(call_id, str)
        or not call_id.startswith(ASK_USER_CALL_ID_PREFIX)
    ):
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    tool_result = cast(dict[str, Any], messages[-1])
    if tool_result.get("tool_call_id") != call_id:
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    response_id, interrupt_calls = _decode_interrupt_cursor(call_id)
    answers = _interrupt_answers(tool_result.get("content"))
    if set(answers) != {f"resume_{index}" for index in range(len(interrupt_calls))}:
        msg = "Open WebUI returned an incomplete interrupt answer batch."
        raise ValueError(msg)
    outputs = []
    for index, interrupt_call in enumerate(interrupt_calls):
        payload = _interrupt_payload(interrupt_call)
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": interrupt_call.call_id,
                "output": _resume_value(answers.get(f"resume_{index}"), payload),
            }
        )
    return outputs, response_id


def _decode_interrupt_cursor(
    call_id: str,
) -> tuple[str, list[ResponseFunctionToolCall]]:
    try:
        encoded = call_id.removeprefix(ASK_USER_CALL_ID_PREFIX)
        padding = "=" * (-len(encoded) % 4)
        cursor = InterruptCursor.model_validate_json(
            base64.urlsafe_b64decode(encoded + padding)
        )
    except (TypeError, ValueError) as exc:
        msg = "Open WebUI returned an invalid interrupt cursor."
        raise ValueError(msg) from exc
    return cursor.previous_response_id, cursor.calls


def _interrupt_answers(content: object) -> dict[str, Any]:
    if content == ASK_USER_REJECTED_OUTPUT:
        raise InterruptCancelled
    try:
        answer = json.loads(content) if isinstance(content, str) else None
    except ValueError as exc:
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg) from exc
    if isinstance(answer, dict) and answer.get("status") == "cancelled":
        raise InterruptCancelled
    answers = answer.get("answers") if isinstance(answer, dict) else None
    if (
        not isinstance(answer, dict)
        or answer.get("status") != "answered"
        or not isinstance(answers, dict)
    ):
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg)
    return answers


def _interrupts_to_ask_user(
    response_id: str,
    calls: list[ResponseFunctionToolCall],
    *,
    streaming: bool = False,
) -> dict[str, Any]:
    """Present one atomic LGOS interrupt batch as one native question card."""
    cursor = InterruptCursor(previous_response_id=response_id, calls=calls)
    questions = [
        _interrupt_question(_interrupt_payload(call), index)
        for index, call in enumerate(cursor.calls)
    ]
    # The Pipe host only returns its native ask-user call and answer on resume.
    # Persist the upstream IDs here so reconnects need no separate state store.
    encoded = cursor.model_dump_json(exclude_none=True).encode()
    result = {
        "id": ASK_USER_CALL_ID_PREFIX
        + base64.urlsafe_b64encode(encoded).decode().rstrip("="),
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL_NAME,
            "arguments": json.dumps(
                {
                    "questions": questions,
                    "allow_other": any(
                        question["allow_other"] for question in questions
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }
    if streaming:
        result["index"] = 0
    return result


def _openwebui_interrupt_chunk(
    model_id: str,
    response_id: str,
    calls: list[ResponseFunctionToolCall],
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-lgos-responses",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        _interrupts_to_ask_user(
                            response_id,
                            calls,
                            streaming=True,
                        )
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _openwebui_interrupt_completion(
    model_id: str,
    response_id: str,
    calls: list[ResponseFunctionToolCall],
    content: str = "",
) -> dict[str, Any]:
    ask_user = _interrupts_to_ask_user(response_id, calls)
    output: list[dict[str, Any]] = []
    if content:
        output.append(
            {
                "type": "message",
                "id": "msg_answer",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    output.append(
        {
            "type": "function_call",
            "id": ask_user["id"],
            "call_id": ask_user["id"],
            "name": ASK_USER_TOOL_NAME,
            "arguments": ask_user["function"]["arguments"],
            "status": "pending",
        }
    )
    return {
        "id": "chatcmpl-lgos-responses",
        "object": "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [ask_user],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "output": output,
    }


def _interrupt_payload(call: ResponseFunctionToolCall) -> dict[str, Any]:
    try:
        arguments = json.loads(call.arguments)
    except (TypeError, ValueError) as exc:
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg) from exc
    if not isinstance(arguments, dict):
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg)
    return arguments


def _interrupt_question(payload: object, index: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        msg = "Open WebUI requires an object interrupt payload."
        raise ValueError(msg)
    question = payload.get("question")
    choices = payload.get("choices")
    allow_other = payload.get("allow_other", False)
    if not isinstance(question, str) or not question.strip():
        msg = "Open WebUI interrupt payload requires a question."
        raise ValueError(msg)
    if (
        not isinstance(choices, list)
        or not 2 <= len(choices) <= 3
        or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
        or len(set(choices)) != len(choices)
        or not isinstance(allow_other, bool)
    ):
        msg = "Open WebUI interrupts require 2-3 unique string choices."
        raise ValueError(msg)

    details = {
        key: value
        for key, value in payload.items()
        if key not in {"question", "choices", "allow_other"}
    }
    prompt = question.strip()
    if details:
        prompt = f"{prompt}\n\n{json.dumps(details, ensure_ascii=False, indent=2)}"
    if len(prompt) > ASK_USER_QUESTION_MAX_LENGTH:
        msg = (
            "Open WebUI interrupt question exceeds "
            f"{ASK_USER_QUESTION_MAX_LENGTH} characters."
        )
        raise ValueError(msg)
    return {
        "id": f"resume_{index}",
        "header": "Human input",
        "question": prompt,
        "options": [
            {"label": choice, "description": f"Resume with {choice!r}."}
            for choice in choices
        ],
        "allow_other": allow_other,
    }


def _resume_value(answer: object, payload: object) -> str:
    if not isinstance(answer, dict) or not isinstance(payload, dict):
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg)
    if answer.get("type") == "option":
        index = answer.get("option_index")
        choices = payload.get("choices")
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and isinstance(choices, list)
            and 0 <= index < len(choices)
            and isinstance(choices[index], str)
        ):
            return choices[index]
    elif answer.get("type") == "other" and payload.get("allow_other") is True:
        text = answer.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    msg = "Open WebUI returned an invalid interrupt answer."
    raise ValueError(msg)
