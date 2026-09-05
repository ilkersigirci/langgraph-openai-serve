"""Adapt Responses interrupt calls to Open WebUI's native ask-user UI."""

import base64
import json
from typing import Any, cast

from openai.types.responses import ResponseFunctionToolCall

from .contracts import (
    ASK_USER_CALL_ID_PREFIX,
    ASK_USER_MAX_QUESTIONS,
    ASK_USER_QUESTION_MAX_LENGTH,
    ASK_USER_REJECTED_OUTPUT,
    ASK_USER_TOOL_NAME,
    INTERRUPT_CANCELLED_MESSAGE,
    INTERRUPT_TOOL_NAME,
    InterruptCancelled,
)


def _ask_user_to_resume(messages: list[Any]) -> list[dict[str, Any]] | None:
    """Restore canonical Responses calls from Open WebUI's persisted answer."""
    if not messages:
        return None

    has_tool_result = (
        len(messages) >= 2
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "tool"
    )
    assistant = messages[-2] if has_tool_result else messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return None

    tool_calls = assistant.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return None
    ask_call = tool_calls[0]
    function = ask_call.get("function") if isinstance(ask_call, dict) else None
    call_id = ask_call.get("id") if isinstance(ask_call, dict) else None
    if (
        not isinstance(function, dict)
        or function.get("name") != ASK_USER_TOOL_NAME
        or not isinstance(call_id, str)
        or not call_id.startswith(ASK_USER_CALL_ID_PREFIX)
    ):
        return None
    if not has_tool_result:
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    tool_result = cast(dict[str, Any], messages[-1])
    if tool_result.get("tool_call_id") != call_id:
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    interrupt_calls = _decode_interrupt_cursor(call_id)
    answers = _interrupt_answers(tool_result.get("content"))
    replay = list(interrupt_calls)
    for index, interrupt_call in enumerate(interrupt_calls):
        payload = _interrupt_payload(interrupt_call)
        replay.append(
            {
                "type": "function_call_output",
                "call_id": interrupt_call["call_id"],
                "output": json.dumps(
                    {"resume": _resume_value(answers.get(f"resume_{index}"), payload)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return replay


def _decode_interrupt_cursor(call_id: str) -> list[dict[str, Any]]:
    try:
        encoded = call_id.removeprefix(ASK_USER_CALL_ID_PREFIX)
        padding = "=" * (-len(encoded) % 4)
        calls = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (TypeError, ValueError) as exc:
        msg = "Open WebUI returned an invalid interrupt cursor."
        raise ValueError(msg) from exc
    if not isinstance(calls, list) or not 1 <= len(calls) <= ASK_USER_MAX_QUESTIONS:
        msg = "Open WebUI returned an invalid interrupt cursor."
        raise ValueError(msg)
    for call in calls:
        if (
            not isinstance(call, dict)
            or call.get("type") != "function_call"
            or call.get("name") != INTERRUPT_TOOL_NAME
            or not isinstance(call.get("call_id"), str)
            or not isinstance(call.get("arguments"), str)
        ):
            msg = "Open WebUI returned an invalid interrupt cursor."
            raise ValueError(msg)
    return cast(list[dict[str, Any]], calls)


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
    calls: list[ResponseFunctionToolCall],
    *,
    streaming: bool = False,
) -> dict[str, Any]:
    """Present one atomic LGOS interrupt batch as one native question card."""
    if len(calls) > ASK_USER_MAX_QUESTIONS:
        msg = f"Open WebUI supports at most {ASK_USER_MAX_QUESTIONS} interrupts per batch."
        raise ValueError(msg)

    replay = []
    questions = []
    for index, call in enumerate(calls):
        if call.name != INTERRUPT_TOOL_NAME:
            msg = "LangGraph API returned a mixed function-call batch."
            raise ValueError(msg)
        item = call.model_dump(mode="json", exclude_none=True)
        replay.append(item)
        questions.append(_interrupt_question(_interrupt_payload(item), index))

    cursor = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).encode()
    result = {
        "id": ASK_USER_CALL_ID_PREFIX
        + base64.urlsafe_b64encode(cursor).decode().rstrip("="),
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
                    "tool_calls": [_interrupts_to_ask_user(calls, streaming=True)]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _openwebui_interrupt_completion(
    model_id: str,
    calls: list[ResponseFunctionToolCall],
) -> dict[str, Any]:
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
                    "content": None,
                    "tool_calls": [_interrupts_to_ask_user(calls)],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _interrupt_payload(call: dict[str, Any]) -> object:
    try:
        arguments = json.loads(call["arguments"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg) from exc
    if not isinstance(arguments, dict) or "payload" not in arguments:
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg)
    return arguments["payload"]


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


def _interrupt_cancelled_response(
    model_id: str,
    *,
    streaming: bool,
) -> dict[str, Any]:
    message = {"role": "assistant", "content": INTERRUPT_CANCELLED_MESSAGE}
    return {
        "id": "chatcmpl-lgos-interrupt-cancelled",
        "object": "chat.completion.chunk" if streaming else "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta" if streaming else "message": message,
                "finish_reason": "stop",
            }
        ],
    }
