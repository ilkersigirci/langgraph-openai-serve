"""Shared function-call decoding for the OpenAI protocol adapters."""

import json

from langchain_core.messages import InvalidToolCall, ToolCall
from langchain_core.messages.tool import invalid_tool_call, tool_call


def decode_function_call(
    *, name: str, arguments: str, call_id: str
) -> ToolCall | InvalidToolCall:
    """Parse once, preserving malformed arguments for the graph to handle."""
    error: str | None = None
    try:
        parsed = json.loads(arguments)
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
