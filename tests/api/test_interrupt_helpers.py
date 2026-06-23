import pytest

from langgraph_openai_serve.api.chat.utils.interrupts import (
    InvalidResumeRequestError,
    parse_resume_value,
)
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage


def message(**kwargs) -> ChatCompletionRequestMessage:
    return ChatCompletionRequestMessage.model_validate(kwargs)


def test_parse_resume_value_accepts_interrupt_tool_response() -> None:
    messages = [
        message(
            role="tool",
            tool_call_id="lg_interrupt_interrupt-1",
            content='{"resume": "yes"}',
        )
    ]

    assert parse_resume_value(messages) == "yes"


def test_parse_resume_value_rejects_invalid_tool_responses() -> None:
    cases = [
        (
            message(role="tool", content='{"resume": "yes"}'),
            "tool_call_id",
        ),
        (
            message(
                role="tool",
                tool_call_id="call_other",
                content='{"resume": "yes"}',
            ),
            "not for a LangGraph interrupt",
        ),
        (
            message(
                role="tool",
                tool_call_id="lg_interrupt_interrupt-1",
                content='{"value": "yes"}',
            ),
            "resume",
        ),
    ]

    for tool_message, error in cases:
        with pytest.raises(InvalidResumeRequestError, match=error):
            parse_resume_value([tool_message])
