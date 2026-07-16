import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequestMessage,
    Role,
    ToolCall as RequestToolCall,
    ToolCallFunction,
)
from langgraph_openai_serve.utils.message import (
    InvalidChatMessageError,
    convert_to_lc_messages,
)


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
        [
            ChatCompletionRequestMessage(
                role=role,
                content=content,
                name="participant",
            )
        ]
    )

    assert len(result) == 1
    assert isinstance(result[0], message_type)
    assert result[0].content == expected_content
    assert result[0].name == "participant"


def test_tool_call_round_trip_preserves_wire_fields_and_order():
    arguments = '{"city":"Istanbul","unit":"celsius"}'
    request_messages = [
        ChatCompletionRequestMessage(role=Role.USER, content="Weather?"),
        ChatCompletionRequestMessage(
            role=Role.ASSISTANT,
            content=None,
            name="planner",
            tool_calls=[
                RequestToolCall(
                    id="call_weather",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=arguments,
                    ),
                )
            ],
        ),
        ChatCompletionRequestMessage(
            role=Role.TOOL,
            content='{"temperature":20}',
            name="get_weather",
            tool_call_id="call_weather",
        ),
    ]

    result = convert_to_lc_messages(request_messages)

    assert [type(message) for message in result] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    assistant_message = result[1]
    assert isinstance(assistant_message, AIMessage)
    assert assistant_message.content == ""
    assert assistant_message.name == "planner"
    assert assistant_message.additional_kwargs["tool_calls"] == [
        {
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": arguments,
            },
        }
    ]
    assert assistant_message.tool_calls == [
        {
            "name": "get_weather",
            "args": {"city": "Istanbul", "unit": "celsius"},
            "id": "call_weather",
            "type": "tool_call",
        }
    ]
    assert assistant_message.invalid_tool_calls == []

    tool_message = result[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == '{"temperature":20}'
    assert tool_message.name == "get_weather"
    assert tool_message.tool_call_id == "call_weather"


@pytest.mark.parametrize(
    "arguments",
    ['{"city":', '"Istanbul"', "[]", "false", "0", "null", '""'],
    ids=(
        "invalid-json",
        "string-json",
        "empty-array-json",
        "false-json",
        "zero-json",
        "null-json",
        "empty-string-json",
    ),
)
def test_invalid_tool_arguments_preserve_raw_call(arguments):
    result = convert_to_lc_messages(
        [
            ChatCompletionRequestMessage(
                role=Role.ASSISTANT,
                tool_calls=[
                    RequestToolCall(
                        id="call_invalid",
                        function=ToolCallFunction(
                            name="get_weather",
                            arguments=arguments,
                        ),
                    )
                ],
            )
        ]
    )

    assistant_message = result[0]
    assert isinstance(assistant_message, AIMessage)
    assert (
        assistant_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
        == arguments
    )
    assert assistant_message.tool_calls == []
    assert len(assistant_message.invalid_tool_calls) == 1
    invalid_tool_call = assistant_message.invalid_tool_calls[0]
    assert invalid_tool_call["id"] == "call_invalid"
    assert invalid_tool_call["name"] == "get_weather"
    assert invalid_tool_call["args"] == arguments
    assert invalid_tool_call["error"]


def test_empty_tool_arguments_support_parameterless_tools():
    result = convert_to_lc_messages(
        [
            ChatCompletionRequestMessage(
                role=Role.ASSISTANT,
                tool_calls=[
                    RequestToolCall(
                        id="call_parameterless",
                        function=ToolCallFunction(
                            name="ping",
                            arguments="",
                        ),
                    )
                ],
            )
        ]
    )

    assistant_message = result[0]
    assert isinstance(assistant_message, AIMessage)
    assert assistant_message.tool_calls == [
        {
            "name": "ping",
            "args": {},
            "id": "call_parameterless",
            "type": "tool_call",
        }
    ]
    assert assistant_message.invalid_tool_calls == []


def test_tool_message_requires_tool_call_id():
    message = ChatCompletionRequestMessage(
        role=Role.TOOL,
        content="Tool result",
    )

    with pytest.raises(InvalidChatMessageError, match="tool_call_id"):
        convert_to_lc_messages([message])


def test_empty_messages_produce_an_empty_list():
    assert convert_to_lc_messages([]) == []
