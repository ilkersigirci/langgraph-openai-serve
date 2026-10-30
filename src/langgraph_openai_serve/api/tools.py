"""Function-call decoding and encoding shared by the OpenAI protocol adapters."""

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, InvalidToolCall, ToolCall
from langchain_core.messages.tool import invalid_tool_call, tool_call

from langgraph_openai_serve.core.errors import GraphError


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """One final assistant function call, as both OpenAI APIs serialize it."""

    call_id: str
    name: str
    arguments: str


def decode_function_call(
    *, name: str, arguments: str, call_id: str
) -> ToolCall | InvalidToolCall:
    """Parse once, preserving malformed arguments for the graph to handle."""
    error: str | None = None
    try:
        # Like LangChain's parse_tool_call, empty arguments call a function
        # that takes no parameters.
        parsed = json.loads(arguments or "{}")
        # json.loads accepts NaN, Infinity, and numeric overflow such as 1e999.
        # None of those may enter an otherwise valid JSON argument object.
        json.dumps(parsed, allow_nan=False)
    except ValueError as exc:
        error = f"Function arguments are not valid JSON: {exc}"
    else:
        if isinstance(parsed, dict):
            return tool_call(name=name, args=parsed, id=call_id)
        error = "Function arguments must decode to a JSON object."
    return invalid_tool_call(name=name, args=arguments, id=call_id, error=error)


def encode_function_calls(
    message: AIMessage, *, include_invalid: bool = False
) -> list[FunctionCall]:
    """
    Serialize the final message's tool calls, or fail the graph's output.

    Invalid calls are served only with ``include_invalid``, for a truncated
    message, and keep their raw arguments.
    """
    if message.invalid_tool_calls and not include_invalid:
        msg = "The final assistant message contains invalid tool calls."
        raise GraphError(msg)
    calls: list[tuple[str | None, str | None, str | None]] = [
        *(
            (call["id"], call["name"], _dump_arguments(call["args"]))
            for call in message.tool_calls
        ),
        *(
            (call["id"], call["name"], call["args"])
            for call in message.invalid_tool_calls
        ),
    ]
    encoded: list[FunctionCall] = []
    for call_id, name, arguments in calls:
        if not call_id or not name or arguments is None:
            msg = "Final assistant tool calls need an id, a name, and arguments."
            raise GraphError(msg)
        encoded.append(FunctionCall(call_id=call_id, name=name, arguments=arguments))
    if len({call.call_id for call in encoded}) != len(encoded):
        msg = "The final assistant message repeats a tool call id."
        raise GraphError(msg)
    return encoded


def _dump_arguments(arguments: dict[str, object]) -> str:
    try:
        return json.dumps(arguments, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        msg = "Final assistant tool call arguments must be valid JSON values."
        raise GraphError(msg) from exc
