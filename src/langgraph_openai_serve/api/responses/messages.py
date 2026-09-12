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
    ResponseCustomToolCallInput,
    ResponseCustomToolCallOutputInput,
    ResponseFunctionCallInput,
    ResponseFunctionCallOutputInput,
    ResponseInputFile,
    ResponseInputItem,
    ResponseInputMessage,
    ResponseInputText,
    ResponseOutputMessageInput,
    ResponseWebSearchCallInput,
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
        if isinstance(item, ResponseWebSearchCallInput):
            messages.append(_web_search_message(item))
            index += 1
            continue
        if isinstance(item, (ResponseFunctionCallInput, ResponseCustomToolCallInput)):
            calls: list[ResponseFunctionCallInput | ResponseCustomToolCallInput] = []
            while index < len(input_value) and isinstance(
                input_value[index],
                (ResponseFunctionCallInput, ResponseCustomToolCallInput),
            ):
                calls.append(
                    cast(
                        "ResponseFunctionCallInput | ResponseCustomToolCallInput",
                        input_value[index],
                    )
                )
                index += 1
            messages.append(_tool_call_message(calls))
            continue
        messages.append(_message_from_item(item))
        index += 1
    return messages


def _validate_replay_ids(
    items: list[ResponseInputItem],
) -> None:
    seen_item_ids: set[str] = set()
    seen_call_ids: dict[str, str] = {}
    seen_output_call_ids: set[str] = set()
    for item in items:
        item_id = getattr(item, "id", None)
        if item_id is not None:
            if item_id in seen_item_ids:
                msg = f"Responses input contains duplicate item id '{item_id}'."
                raise InvalidResponsesInputError(msg)
            seen_item_ids.add(item_id)

        if isinstance(item, (ResponseFunctionCallInput, ResponseCustomToolCallInput)):
            if item.call_id in seen_call_ids:
                msg = f"Responses input contains duplicate call_id '{item.call_id}'."
                raise InvalidResponsesInputError(msg)
            seen_call_ids[item.call_id] = item.type
        elif isinstance(
            item, (ResponseFunctionCallOutputInput, ResponseCustomToolCallOutputInput)
        ):
            if item.call_id in seen_output_call_ids:
                msg = (
                    "Responses input contains duplicate tool output call_id "
                    f"'{item.call_id}'."
                )
                raise InvalidResponsesInputError(msg)
            if item.type != f"{seen_call_ids.get(item.call_id)}_output":
                msg = (
                    "Responses tool output call_id and type must match an earlier "
                    f"tool call; got '{item.call_id}'."
                )
                raise InvalidResponsesInputError(msg)
            seen_output_call_ids.add(item.call_id)

    unanswered = seen_call_ids.keys() - seen_output_call_ids
    if unanswered:
        msg = (
            "Responses tool calls require matching tool output items; "
            f"missing outputs for {', '.join(sorted(unanswered))}."
        )
        raise InvalidResponsesInputError(msg)


def _message_from_item(item: ResponseInputItem) -> BaseMessage:
    if isinstance(item, ResponseOutputMessageInput):
        return AIMessage(
            id=item.id,
            content=_output_content(item),
            additional_kwargs={"id": item.id, "phase": item.phase},
            response_metadata={"model_provider": "openai"},
        )
    if isinstance(
        item, (ResponseFunctionCallOutputInput, ResponseCustomToolCallOutputInput)
    ):
        return ToolMessage(
            id=item.id,
            content=(
                [{"type": "custom_tool_call_output", "output": item.output}]
                if isinstance(item, ResponseCustomToolCallOutputInput)
                else item.output
            ),
            tool_call_id=item.call_id,
        )
    if isinstance(item, (ResponseFunctionCallInput, ResponseCustomToolCallInput)):
        msg = "Tool calls must be grouped before message conversion."
        raise TypeError(msg)
    if not isinstance(item, ResponseInputMessage):
        msg = "Web-search calls must be converted before message conversion."
        raise TypeError(msg)

    content = _input_content(item.content)
    match item.role:
        case "assistant":
            return AIMessage(
                content=[
                    {**cast("dict[str, Any]", part), "phase": item.phase}
                    for part in (
                        [{"type": "text", "text": content}]
                        if isinstance(content, str)
                        else content
                    )
                ],
                additional_kwargs={"phase": item.phase},
                response_metadata={"model_provider": "openai"},
            )
        case "user":
            return HumanMessage(content=content)
        case "developer":
            return SystemMessage(
                content=content,
                additional_kwargs={"__openai_role__": "developer"},
            )
        case "system":
            return SystemMessage(content=content)


def _web_search_message(item: ResponseWebSearchCallInput) -> AIMessage:
    return AIMessage(
        id=item.id,
        content=[item.model_dump(mode="json", exclude_none=True)],
        response_metadata={"model_provider": "openai"},
    )


def _tool_call_message(
    calls: list[ResponseFunctionCallInput | ResponseCustomToolCallInput],
) -> AIMessage:
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []
    for call in calls:
        if isinstance(call, ResponseCustomToolCallInput):
            tool_calls.append(
                ToolCall(
                    name=call.name,
                    args={"__arg1": call.input},
                    id=call.call_id,
                    type="tool_call",
                )
            )
            continue
        parsed = decode_function_call(
            name=call.name, arguments=call.arguments, call_id=call.call_id
        )
        if parsed["type"] == "tool_call":
            tool_calls.append(parsed)
        else:
            invalid_tool_calls.append(parsed)

    return AIMessage(
        # LangChain's Responses adapter reads item IDs from content blocks;
        # tool_calls alone preserves call_id but loses the distinct item id.
        content=[call.model_dump(mode="json", exclude_none=True) for call in calls],
        response_metadata={"model_provider": "openai"},
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
    item: ResponseOutputMessageInput,
) -> list[str | dict[Any, Any]]:
    return cast(
        "list[str | dict[Any, Any]]",
        [
            {
                **part.model_dump(mode="json", exclude_none=True),
                "type": "text" if part.type == "output_text" else part.type,
                "id": item.id,
                "phase": item.phase,
            }
            for part in item.content
        ],
    )


__all__ = ["InvalidResponsesInputError", "convert_responses_input"]
