from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage, Role
from langgraph_openai_serve.utils.message import convert_to_lc_messages


def test_empty_messages_list():
    """Test conversion with an empty list of messages."""
    messages = []
    result = convert_to_lc_messages(messages)
    assert result == []
    assert isinstance(result, list)


def test_system_message_conversion():
    """Test conversion of system messages."""
    messages = [
        ChatCompletionRequestMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
    ]
    result = convert_to_lc_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are a helpful assistant."


def test_user_message_conversion():
    """Test conversion of user messages."""
    messages = [
        ChatCompletionRequestMessage(role=Role.USER, content="Hello, how are you?")
    ]
    result = convert_to_lc_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello, how are you?"


def test_assistant_message_conversion():
    """Test conversion of assistant messages."""
    messages = [
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content="I'm doing well, thank you!")
    ]
    result = convert_to_lc_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "I'm doing well, thank you!"


def test_mixed_message_conversion():
    """Test conversion of mixed message types."""
    messages = [
        ChatCompletionRequestMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatCompletionRequestMessage(role=Role.USER, content="Hello, how are you?"),
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content="I'm doing well, thank you!")
    ]
    result = convert_to_lc_messages(messages)

    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are a helpful assistant."
    assert isinstance(result[1], HumanMessage)
    assert result[1].content == "Hello, how are you?"
    assert isinstance(result[2], AIMessage)
    assert result[2].content == "I'm doing well, thank you!"


def test_none_content_handling():
    """Test conversion of messages with None content."""
    messages = [
        ChatCompletionRequestMessage(role=Role.SYSTEM, content=None),
        ChatCompletionRequestMessage(role=Role.USER, content=None),
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content=None)
    ]
    result = convert_to_lc_messages(messages)

    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == ""
    assert isinstance(result[1], HumanMessage)
    assert result[1].content == ""
    assert isinstance(result[2], AIMessage)
    assert result[2].content == ""


def test_unsupported_role_types():
    """Test that unsupported role types (function, tool) are skipped."""
    messages = [
        ChatCompletionRequestMessage(role=Role.SYSTEM, content="System message"),
        ChatCompletionRequestMessage(role=Role.FUNCTION, content="Function content"),
        ChatCompletionRequestMessage(role=Role.TOOL, content="Tool content"),
        ChatCompletionRequestMessage(role=Role.USER, content="User message"),
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content="Assistant message")
    ]
    result = convert_to_lc_messages(messages)

    # Only system, user, and assistant messages should be converted
    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "System message"
    assert isinstance(result[1], HumanMessage)
    assert result[1].content == "User message"
    assert isinstance(result[2], AIMessage)
    assert result[2].content == "Assistant message"
