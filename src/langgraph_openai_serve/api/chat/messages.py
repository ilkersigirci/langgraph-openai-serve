"""Convert Chat Completions messages into LangChain messages."""

from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionMessageContent,
    ChatCompletionRequestMessage,
    Role,
)
from langgraph_openai_serve.api.tools import decode_function_call


class InvalidChatMessageError(ValueError):
    """Raised when a chat message is missing a role-specific required field."""


def _langchain_content(
    content: ChatCompletionMessageContent | None,
) -> str | list[str | dict[Any, Any]]:
    """Pass OpenAI content parts through LangChain's compatible message type."""
    return cast("str | list[str | dict[Any, Any]]", content or "")


def convert_to_lc_messages(
    messages: list[ChatCompletionRequestMessage],
) -> list[BaseMessage]:
    """
    Convert OpenAI messages to LangChain messages.

    This function converts a list of OpenAI-compatible message objects to their
    LangChain equivalents for use with LangGraph.

    Args:
        messages: A list of OpenAI chat completion request messages to convert.

    Returns:
        A list of LangChain message objects.

    """
    lc_messages: list[BaseMessage] = []
    for m in messages:
        match m.role:
            case Role.SYSTEM:
                lc_messages.append(
                    SystemMessage(content=_langchain_content(m.content), name=m.name)
                )
            case Role.USER:
                lc_messages.append(
                    HumanMessage(content=_langchain_content(m.content), name=m.name)
                )
            case Role.ASSISTANT:
                lc_messages.append(_assistant_message(m))
            case Role.TOOL:
                if m.tool_call_id is None:
                    msg = "Tool messages require the 'tool_call_id' field."
                    raise InvalidChatMessageError(msg)
                lc_messages.append(
                    ToolMessage(
                        content=_langchain_content(m.content),
                        name=m.name,
                        tool_call_id=m.tool_call_id,
                    )
                )
    return lc_messages


def _assistant_message(message: ChatCompletionRequestMessage) -> AIMessage:
    """Preserve OpenAI assistant fields and expose parsed LangChain tool calls."""
    additional_kwargs: dict[str, Any] = {}

    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []
    if message.tool_calls is not None:
        raw_tool_calls = [
            tool_call.model_dump(mode="json") for tool_call in message.tool_calls
        ]
        additional_kwargs["tool_calls"] = raw_tool_calls

        for call in message.tool_calls:
            parsed = decode_function_call(
                name=call.function.name,
                arguments=call.function.arguments or "{}",
                call_id=call.id,
            )
            if parsed["type"] == "tool_call":
                tool_calls.append(parsed)
            else:
                invalid_tool_calls.append(parsed)

    return AIMessage(
        content=_langchain_content(message.content),
        name=message.name,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )
