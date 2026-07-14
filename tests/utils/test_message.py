import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage, Role
from langgraph_openai_serve.utils.message import convert_to_lc_messages


@pytest.mark.parametrize(
    ("role", "content", "message_type", "expected_content"),
    [
        (Role.SYSTEM, "You are helpful.", SystemMessage, "You are helpful."),
        (Role.USER, "Hello", HumanMessage, "Hello"),
        (Role.ASSISTANT, "Hi there", AIMessage, "Hi there"),
        (Role.SYSTEM, None, SystemMessage, ""),
        (Role.USER, None, HumanMessage, ""),
        (Role.ASSISTANT, None, AIMessage, ""),
    ],
    ids=(
        "system-text",
        "user-text",
        "assistant-text",
        "system-null",
        "user-null",
        "assistant-null",
    ),
)
def test_supported_message_conversion(role, content, message_type, expected_content):
    result = convert_to_lc_messages(
        [ChatCompletionRequestMessage(role=role, content=content)]
    )

    assert len(result) == 1
    assert isinstance(result[0], message_type)
    assert result[0].content == expected_content


def test_unsupported_messages_are_removed_without_reordering():
    messages = [
        ChatCompletionRequestMessage(role=Role.SYSTEM, content="System message"),
        ChatCompletionRequestMessage(role=Role.FUNCTION, content="Function content"),
        ChatCompletionRequestMessage(role=Role.TOOL, content="Tool content"),
        ChatCompletionRequestMessage(role=Role.USER, content="User message"),
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content="Assistant message"),
    ]

    result = convert_to_lc_messages(messages)

    assert [(type(message), message.content) for message in result] == [
        (SystemMessage, "System message"),
        (HumanMessage, "User message"),
        (AIMessage, "Assistant message"),
    ]


def test_empty_messages_produce_an_empty_list():
    assert convert_to_lc_messages([]) == []
