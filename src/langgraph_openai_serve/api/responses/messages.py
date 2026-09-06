"""Convert Responses message input into LangChain messages."""

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

from langgraph_openai_serve.api.responses.schemas import (
    ResponseAssistantInputMessage,
    ResponseFunctionCallInput,
    ResponseFunctionCallOutputInput,
    ResponseInputFile,
    ResponseInputItem,
    ResponseInputText,
    ResponseOutputMessageInput,
    ResponseOutputTextInput,
)
from langgraph_openai_serve.api.tools import decode_function_call


class InvalidResponsesInputError(ValueError):
    """Raised when Responses input items cannot be replayed unambiguously."""


def convert_responses_input(
    input_value: str | list[ResponseInputItem],
    *,
    instructions: str | None,
) -> list[BaseMessage]:
    """Normalize supported Responses text and message input."""
    messages: list[BaseMessage] = []
    if instructions is not None:
        messages.append(SystemMessage(content=instructions))

    if isinstance(input_value, str):
        messages.append(HumanMessage(content=input_value))
        return messages

    _validate_replay_ids(input_value)
    index = 0
    while index < len(input_value):
        item = input_value[index]
        if isinstance(item, ResponseFunctionCallInput):
            calls: list[ResponseFunctionCallInput] = []
            while index < len(input_value) and isinstance(
                input_value[index],
                ResponseFunctionCallInput,
            ):
                calls.append(cast("ResponseFunctionCallInput", input_value[index]))
                index += 1
            messages.append(_function_call_message(calls))
            continue
        messages.append(_message_from_item(item))
        index += 1
    return messages


def _validate_replay_ids(
    items: list[ResponseInputItem],
) -> None:
    seen_item_ids: set[str] = set()
    seen_call_ids: set[str] = set()
    seen_output_call_ids: set[str] = set()
    for item in items:
        item_id = getattr(item, "id", None)
        if item_id is not None:
            if item_id in seen_item_ids:
                msg = f"Responses input contains duplicate item id '{item_id}'."
                raise InvalidResponsesInputError(msg)
            seen_item_ids.add(item_id)

        if isinstance(item, ResponseFunctionCallInput):
            if item.call_id in seen_call_ids:
                msg = f"Responses input contains duplicate call_id '{item.call_id}'."
                raise InvalidResponsesInputError(msg)
            seen_call_ids.add(item.call_id)
        elif isinstance(item, ResponseFunctionCallOutputInput):
            if item.call_id in seen_output_call_ids:
                msg = (
                    "Responses input contains duplicate function output call_id "
                    f"'{item.call_id}'."
                )
                raise InvalidResponsesInputError(msg)
            if item.call_id not in seen_call_ids:
                msg = (
                    "Responses function output call_id must match an earlier "
                    f"function call; got '{item.call_id}'."
                )
                raise InvalidResponsesInputError(msg)
            seen_output_call_ids.add(item.call_id)

    unanswered = seen_call_ids - seen_output_call_ids
    if unanswered:
        msg = (
            "Responses function calls require matching function_call_output items; "
            f"missing outputs for {', '.join(sorted(unanswered))}."
        )
        raise InvalidResponsesInputError(msg)


def _message_from_item(item: ResponseInputItem) -> BaseMessage:
    if isinstance(item, ResponseOutputMessageInput):
        return AIMessage(
            id=item.id,
            content=_output_content(item.content),
            additional_kwargs={"id": item.id, "phase": item.phase},
        )
    if isinstance(item, ResponseFunctionCallOutputInput):
        return ToolMessage(content=item.output, tool_call_id=item.call_id)
    if isinstance(item, ResponseFunctionCallInput):
        msg = "Function calls must be grouped before message conversion."
        raise TypeError(msg)

    content = _input_content(item.content)
    if isinstance(item, ResponseAssistantInputMessage):
        return AIMessage(
            content=content,
            additional_kwargs={"phase": item.phase},
        )
    if item.role == "user":
        return HumanMessage(content=content)
    if item.role == "developer":
        return SystemMessage(
            content=content,
            additional_kwargs={"__openai_role__": "developer"},
        )
    return SystemMessage(content=content)


def _function_call_message(calls: list[ResponseFunctionCallInput]) -> AIMessage:
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []
    for call in calls:
        parsed = decode_function_call(
            name=call.name, arguments=call.arguments, call_id=call.call_id
        )
        if parsed["type"] == "tool_call":
            tool_calls.append(parsed)
        else:
            invalid_tool_calls.append(parsed)

    return AIMessage(
        content="",
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )


def _input_content(
    content: str | list[ResponseInputText | ResponseInputFile],
) -> str | list[str | dict[Any, Any]]:
    if isinstance(content, str):
        return content
    return cast(
        "list[str | dict[Any, Any]]",
        [
            (
                {"type": "text", "text": part.text}
                if isinstance(part, ResponseInputText)
                else {"type": "file", "file": {"file_id": part.file_id}}
            )
            for part in content
        ],
    )


def _output_content(
    content: list[ResponseOutputTextInput],
) -> list[str | dict[Any, Any]]:
    return cast(
        "list[str | dict[Any, Any]]",
        [{"type": "text", "text": part.text} for part in content],
    )


__all__ = ["InvalidResponsesInputError", "convert_responses_input"]
