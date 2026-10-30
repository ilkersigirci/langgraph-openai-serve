"""Adapt Responses interrupt calls to Open WebUI's native ask-user UI."""

import json
from collections.abc import Sequence
from typing import Annotated, Any

from openai.types.responses import ResponseFunctionToolCall
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
)

from .contracts import (
    ASK_USER_CALL_ID_PREFIX,
    ASK_USER_LABEL_MAX_LENGTH,
    ASK_USER_MAX_QUESTIONS,
    ASK_USER_QUESTION_ID_MAX_LENGTH,
    ASK_USER_QUESTION_MAX_LENGTH,
    ASK_USER_REJECTED_OUTPUT,
    ASK_USER_TOOL_NAME,
    AskUserResult,
    InterruptCancelled,
    OpenWebUIMessage,
)
from .responses import _openwebui_chunk

ASK_USER_MIN_CHOICES = 2
ASK_USER_MAX_CHOICES = 3


class _InterruptPayload(BaseModel):
    """The interrupt arguments Open WebUI's native ask-user card can represent."""

    model_config = ConfigDict(strict=True)

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    choices: list[str] = Field(
        min_length=ASK_USER_MIN_CHOICES,
        max_length=ASK_USER_MAX_CHOICES,
    )
    allow_other: bool = False

    @field_validator("choices")
    @classmethod
    def valid_choices(cls, choices: list[str]) -> list[str]:
        # Answers carry the option label, which Open WebUI strips and truncates.
        if len(set(choices)) != len(choices) or any(
            not choice or choice.strip()[:ASK_USER_LABEL_MAX_LENGTH] != choice
            for choice in choices
        ):
            msg = "Choices must be unique labels that Open WebUI keeps unchanged."
            raise ValueError(msg)
        return choices


def _ask_user_to_resume(
    messages: Sequence[OpenWebUIMessage],
) -> tuple[list[dict[str, str]], str] | None:
    """Restore a Responses continuation from Open WebUI's persisted answer."""
    assistant_index = len(messages) - 1
    while assistant_index >= 0 and messages[assistant_index].role == "tool":
        assistant_index -= 1
    if assistant_index < 0:
        return None
    assistant = messages[assistant_index]
    if assistant.role != "assistant":
        return None

    tool_calls = assistant.tool_calls
    if not any(call.id.startswith(ASK_USER_CALL_ID_PREFIX) for call in tool_calls):
        return None
    # An incomplete exchange must fail rather than start a new graph run.
    card = tool_calls[0]
    if (
        len(tool_calls) != 1
        or assistant_index != len(messages) - 2
        or card.function.name != ASK_USER_TOOL_NAME
        or messages[-1].tool_call_id != card.id
    ):
        msg = "Open WebUI returned an incomplete interrupt batch."
        raise ValueError(msg)

    # Question IDs are the LGOS call IDs; LGOS checks that the batch is complete.
    result = _ask_user_result(messages[-1].content)
    outputs = [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": answer.output(),
        }
        for call_id, answer in result.answers.items()
    ]
    return outputs, card.id.removeprefix(ASK_USER_CALL_ID_PREFIX)


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
    """
    Present one LGOS interrupt batch as one native question card.

    Returns the ``ask_user`` call and the question details the card cannot
    show, to show above it.
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
        # The call ID carries the paused Response ID, which
        # _ask_user_to_resume reads back from the stored answer.
        "id": f"{ASK_USER_CALL_ID_PREFIX}{response_id}",
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL_NAME,
            "arguments": json.dumps(
                {"questions": questions}, ensure_ascii=False, separators=(",", ":")
            ),
        },
    }
    return ask_user, "\n\n".join(reviews)


def _openwebui_interrupt_chunks(
    model_id: str,
    response_id: str,
    calls: list[ResponseFunctionToolCall],
    *,
    after_text: bool = False,
) -> list[dict[str, Any]]:
    ask_user, review = _ask_user_card(response_id, calls)
    chunks = []
    if review:
        # Like a model, stream the text before the call: Open WebUI places a
        # chunk's tool call ahead of its text, which would put the card above
        # the details it asks about.
        content = f"\n\n{review}" if after_text else review
        chunks.append(_openwebui_chunk(model_id, {"content": content}))
    chunks.append(
        _openwebui_chunk(
            model_id,
            {"tool_calls": [{"index": 0, **ask_user}]},
            finish_reason="tool_calls",
        )
    )
    return chunks


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
    # Answers are keyed by question ID, which Open WebUI truncates.
    if len(call.call_id) > ASK_USER_QUESTION_ID_MAX_LENGTH:
        msg = "LangGraph API returned an interrupt call ID Open WebUI cannot keep."
        raise ValueError(msg)
    try:
        payload = _InterruptPayload.model_validate_json(call.arguments)
    except ValidationError as exc:
        msg = (
            "Open WebUI interrupts require a question and "
            f"{ASK_USER_MIN_CHOICES}-{ASK_USER_MAX_CHOICES} unique choices of at "
            f"most {ASK_USER_LABEL_MAX_LENGTH} characters without surrounding spaces."
        )
        raise ValueError(msg) from exc

    # The question is the whole prompt; other payload fields are for programs.
    # Open WebUI's card shows one plain-text line and truncates it, so, as a
    # model would, the card asks the question's first paragraph and the
    # details go above it as Markdown text.
    ask, _, review = payload.question.partition("\n\n")
    if "\n" in ask or len(ask) > ASK_USER_QUESTION_MAX_LENGTH:
        ask, review = "Review the request above before choosing.", payload.question
    return {
        "id": call.call_id,
        "header": "Human input",
        "question": ask,
        "options": [
            {"label": choice, "description": f"Resume with {choice!r}."}
            for choice in payload.choices
        ],
        "allow_other": payload.allow_other,
    }, review
