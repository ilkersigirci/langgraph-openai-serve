import json
from typing import Any, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage, Role


class InvalidChatMessageError(ValueError):
    """Raised when a chat message is missing a role-specific required field."""


def convert_to_lc_messages(
    messages: list[ChatCompletionRequestMessage],
) -> list[BaseMessage]:
    """Convert OpenAI messages to LangChain messages.

    This function converts a list of OpenAI-compatible message objects to their
    LangChain equivalents for use with LangGraph.

    Args:
        messages: A list of OpenAI chat completion request messages to convert.

    Returns:
        A list of LangChain message objects.
    """

    lc_messages: list[BaseMessage] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            lc_messages.append(SystemMessage(content=m.content or "", name=m.name))
        elif m.role == Role.USER:
            lc_messages.append(HumanMessage(content=m.content or "", name=m.name))
        elif m.role == Role.ASSISTANT:
            lc_messages.append(_assistant_message(m))
        elif m.role == Role.TOOL:
            if m.tool_call_id is None:
                raise InvalidChatMessageError(
                    "Tool messages require the 'tool_call_id' field."
                )
            lc_messages.append(
                ToolMessage(
                    content=m.content or "",
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

        for raw_tool_call in raw_tool_calls:
            raw_arguments = raw_tool_call["function"]["arguments"]
            if raw_arguments:
                try:
                    decoded_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    # Preserve LangChain's richer invalid-tool-call diagnostics.
                    pass
                else:
                    if not isinstance(decoded_arguments, dict):
                        invalid_tool_calls.append(
                            make_invalid_tool_call(
                                raw_tool_call,
                                "Tool arguments must decode to a JSON object.",
                            )
                        )
                        continue

            try:
                parsed_tool_call = parse_tool_call(raw_tool_call, return_id=True)
            except OutputParserException as exc:
                invalid_tool_calls.append(
                    make_invalid_tool_call(raw_tool_call, str(exc))
                )
                continue

            if parsed_tool_call is None:
                continue
            if not isinstance(parsed_tool_call.get("args"), dict):
                invalid_tool_calls.append(
                    make_invalid_tool_call(
                        raw_tool_call,
                        "Tool arguments must decode to a JSON object.",
                    )
                )
                continue
            tool_calls.append(cast(ToolCall, parsed_tool_call))

    return AIMessage(
        content=message.content or "",
        name=message.name,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )
