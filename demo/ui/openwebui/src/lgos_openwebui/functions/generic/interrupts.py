"""Adapt Responses interrupt calls to Open WebUI's native ask-user UI."""

import json
from collections.abc import Sequence
from typing import Any, Literal

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, Field, JsonValue

from .contracts import (
    ASK_USER_CALL_ID_PREFIX,
    ASK_USER_LABEL_MAX_LENGTH,
    ASK_USER_MAX_QUESTIONS,
    ASK_USER_QUESTION_ID_MAX_LENGTH,
    ASK_USER_QUESTION_MAX_LENGTH,
    ASK_USER_REJECTED_OUTPUT,
    ASK_USER_TOOL_NAME,
    InterruptCancelled,
    OpenWebUIMessage,
)
from .responses import _openwebui_chunk


class AskUserAnswer(BaseModel):
    """One browser answer: the chosen option's label or free-form text."""

    type: Literal["option", "other"]
    label: str = ""
    text: str = ""


class AskUserResult(BaseModel):
    """The tool result Open WebUI stores when the user answers the card."""

    status: Literal["answered", "cancelled"]
    answers: dict[str, AskUserAnswer] = Field(default_factory=dict)


def _ask_user_to_resume(
    messages: Sequence[OpenWebUIMessage],
) -> tuple[list[dict[str, str]], str] | None:
    """Restore a Responses continuation from Open WebUI's persisted answer."""
    if not messages:
        return None

    assistant_index = len(messages) - 1
    while assistant_index >= 0 and messages[assistant_index].role == "tool":
        assistant_index -= 1
    if assistant_index < 0:
        return None
    assistant = messages[assistant_index]
    if assistant.role != "assistant":
        return None

    tool_calls = assistant.tool_calls
    if not any(
        call.id is not None and call.id.startswith(ASK_USER_CALL_ID_PREFIX)
        for call in tool_calls
    ):
        return None
    # An incomplete exchange must fail rather than start a new graph run.
    function = tool_calls[0].function
    card_id = tool_calls[0].id
    if (
        len(tool_calls) != 1
        or assistant_index != len(messages) - 2
        or function is None
        or function.name != ASK_USER_TOOL_NAME
        or card_id is None
        or messages[-1].tool_call_id != card_id
    ):
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    # Question IDs are the LGOS call IDs; LGOS checks that the batch is complete.
    result = _ask_user_result(messages[-1].content)
    outputs = [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": answer.label if answer.type == "option" else answer.text.strip(),
        }
        for call_id, answer in result.answers.items()
    ]
    return outputs, card_id.removeprefix(ASK_USER_CALL_ID_PREFIX)


def _ask_user_result(content: JsonValue) -> AskUserResult:
    if content == ASK_USER_REJECTED_OUTPUT:
        raise InterruptCancelled
    try:
        result = AskUserResult.model_validate_json(str(content))
    except ValueError as exc:
        msg = "Open WebUI returned an invalid interrupt answer."
        raise ValueError(msg) from exc
    if result.status == "cancelled":
        raise InterruptCancelled
    return result


def _ask_user_card(
    response_id: str,
    calls: Sequence[ResponseFunctionToolCall],
) -> tuple[dict[str, Any], str]:
    """Present one LGOS interrupt batch as one native question card.

    Returns the ``ask_user`` call and the complete text of questions too long
    for the card, to show above it.
    """
    if len(calls) > ASK_USER_MAX_QUESTIONS:
        msg = f"Open WebUI asks at most {ASK_USER_MAX_QUESTIONS} questions at once."
        raise ValueError(msg)
    questions = []
    reviews = []
    for call in calls:
        question, review = _interrupt_question(call)
        questions.append(question)
        if review:
            reviews.append(review)
    ask_user = {
        # Open WebUI keeps this ID with the answer, keyed by question ID.
        "id": f"{ASK_USER_CALL_ID_PREFIX}{response_id}",
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
    return ask_user, "\n\n".join(reviews)


def _openwebui_interrupt_chunk(
    model_id: str,
    response_id: str,
    calls: list[ResponseFunctionToolCall],
) -> dict[str, Any]:
    ask_user, review = _ask_user_card(response_id, calls)
    delta: dict[str, Any] = {"tool_calls": [{"index": 0, **ask_user}]}
    if review:
        delta["content"] = review
    return _openwebui_chunk(model_id, delta, finish_reason="tool_calls")


def _openwebui_interrupt_completion(
    model_id: str,
    response_id: str,
    calls: list[ResponseFunctionToolCall],
    content: str = "",
) -> dict[str, Any]:
    ask_user, review = _ask_user_card(response_id, calls)
    if review:
        content = f"{content}\n\n{review}".strip()
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


def _interrupt_question(call: ResponseFunctionToolCall) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(call.arguments)
    except ValueError as exc:
        msg = "LangGraph API returned invalid interrupt tool arguments."
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = "Open WebUI requires an object interrupt payload."
        raise ValueError(msg)
    question = payload.get("question")
    choices = payload.get("choices")
    allow_other = payload.get("allow_other", False)
    if not isinstance(question, str) or not question.strip():
        msg = "Open WebUI interrupt payload requires a question."
        raise ValueError(msg)
    # Answers carry the option label, which Open WebUI strips and truncates.
    if (
        not isinstance(choices, list)
        or not 2 <= len(choices) <= 3
        or any(
            not isinstance(choice, str)
            or not choice
            or choice.strip()[:ASK_USER_LABEL_MAX_LENGTH] != choice
            for choice in choices
        )
        or len(set(choices)) != len(choices)
        or not isinstance(allow_other, bool)
    ):
        msg = (
            "Open WebUI interrupts require 2-3 unique choices of at most "
            f"{ASK_USER_LABEL_MAX_LENGTH} characters without surrounding spaces."
        )
        raise ValueError(msg)
    # Answers are keyed by question ID, which Open WebUI truncates.
    if len(call.call_id) > ASK_USER_QUESTION_ID_MAX_LENGTH:
        msg = "LangGraph API returned an interrupt call ID Open WebUI cannot keep."
        raise ValueError(msg)

    question = question.strip()
    details = {
        key: value
        for key, value in payload.items()
        if key not in {"question", "choices", "allow_other"}
    }
    encoded = json.dumps(details, ensure_ascii=False, indent=2)
    prompt = f"{question}\n\n{encoded}" if details else question
    review = ""
    if len(prompt) > ASK_USER_QUESTION_MAX_LENGTH:
        # Open WebUI truncates the card text; keep the complete review visible.
        review = f"{question}\n\n```json\n{encoded}\n```" if details else question
        prompt = f"{question}\n\nReview the full request above before choosing."
    return {
        "id": call.call_id,
        "header": "Human input",
        "question": prompt,
        "options": [
            {"label": choice, "description": f"Resume with {choice!r}."}
            for choice in choices
        ],
        "allow_other": allow_other,
    }, review
