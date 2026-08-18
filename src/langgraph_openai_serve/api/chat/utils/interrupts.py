"""OpenAI Chat Completions codec for LangGraph interrupts."""

import json
from dataclasses import dataclass
from typing import Any

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequestMessage,
    Role,
    ToolCall,
)

INTERRUPT_TOOL_NAME = "langgraph_interrupt"
INTERRUPT_TOOL_CALL_ID_PREFIX = "lg_interrupt_"
_INTERRUPT_ARGUMENT_FIELDS = {"run_id", "state_token", "payload"}


class InvalidResumeRequestError(ValueError):
    """Raised when an OpenAI tool exchange is not a valid interrupt resume."""


class InvalidInterruptPayloadError(ValueError):
    """Raised when graph-authored interrupt data cannot cross the JSON API."""


@dataclass(frozen=True)
class InterruptResume:
    """A complete, causally bound set of interrupt answers."""

    run_id: str
    state_token: str
    values: dict[str, Any]


@dataclass(frozen=True)
class _InterruptCall:
    interrupt_id: str
    run_id: str
    state_token: str


def interrupt_tool_call_id(interrupt_id: str) -> str:
    """Format interrupt tool call ID."""
    return f"{INTERRUPT_TOOL_CALL_ID_PREFIX}{interrupt_id}"


def interrupt_arguments(
    *,
    run_id: str,
    state_token: str,
    payload: Any,
) -> str:
    """Encode one interrupt without coercing unsupported graph values."""
    return _dump_json(
        {
            "run_id": run_id,
            "state_token": state_token,
            "payload": payload,
        }
    )


def validate_interrupt_payload(payload: Any) -> None:
    """Reject graph values that cannot cross the OpenAI JSON boundary."""
    _dump_json(payload)


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        msg = "LangGraph interrupt payloads must be valid JSON values."
        raise InvalidInterruptPayloadError(msg) from exc


def parse_resume_request(
    messages: list[ChatCompletionRequestMessage],
) -> InterruptResume | None:
    """
    Parse the trailing canonical assistant/tool interrupt exchange.

    Ordinary tool messages remain ordinary graph input. A LangGraph resume is
    recognized only when the tool results answer a preceding assistant message
    whose function calls are all ``langgraph_interrupt`` calls.
    """
    tool_start = _trailing_tool_start(messages)
    if tool_start is None:
        return None

    tool_messages = messages[tool_start:]
    assistant = messages[tool_start - 1] if tool_start > 0 else None

    if assistant is None or assistant.role != Role.ASSISTANT:
        if any(_is_interrupt_tool_result(message) for message in tool_messages):
            msg = "Interrupt tool results must follow their assistant tool calls."
            raise InvalidResumeRequestError(msg)
        return None

    calls = assistant.tool_calls or []
    interrupt_calls = [
        call for call in calls if call.function.name == INTERRUPT_TOOL_NAME
    ]

    if not interrupt_calls:
        if any(_is_interrupt_tool_result(message) for message in tool_messages):
            msg = "Interrupt tool results must follow their assistant tool calls."
            raise InvalidResumeRequestError(msg)
        return None

    if len(interrupt_calls) != len(calls):
        msg = "Interrupt and ordinary tool calls cannot be resumed in one exchange."
        raise InvalidResumeRequestError(msg)

    parsed_calls = _parse_interrupt_calls(interrupt_calls)
    values = _parse_tool_results(tool_messages, parsed_calls)

    run_ids = {call.run_id for call in parsed_calls.values()}
    state_tokens = {call.state_token for call in parsed_calls.values()}
    if len(run_ids) != 1 or len(state_tokens) != 1:
        msg = (
            "All interrupt tool calls in one exchange must belong to the same run "
            "and interrupt generation."
        )
        raise InvalidResumeRequestError(msg)

    return InterruptResume(
        run_id=run_ids.pop(),
        state_token=state_tokens.pop(),
        values=values,
    )


def _trailing_tool_start(
    messages: list[ChatCompletionRequestMessage],
) -> int | None:
    if not messages or messages[-1].role != Role.TOOL:
        return None

    index = len(messages) - 1
    while index > 0 and messages[index - 1].role == Role.TOOL:
        index -= 1
    return index


def _is_interrupt_tool_result(message: ChatCompletionRequestMessage) -> bool:
    return bool(
        message.tool_call_id
        and message.tool_call_id.startswith(INTERRUPT_TOOL_CALL_ID_PREFIX)
    )


def _parse_interrupt_calls(calls: list[ToolCall]) -> dict[str, _InterruptCall]:
    parsed: dict[str, _InterruptCall] = {}
    for call in calls:
        if call.id in parsed:
            msg = "Interrupt assistant tool_call IDs must be unique."
            raise InvalidResumeRequestError(msg)
        parsed[call.id] = _parse_interrupt_call(call)
    return parsed


def _parse_interrupt_call(call: ToolCall) -> _InterruptCall:
    if not call.id.startswith(INTERRUPT_TOOL_CALL_ID_PREFIX):
        msg = "Interrupt assistant tool_call ID is invalid."
        raise InvalidResumeRequestError(msg)

    interrupt_id = call.id.removeprefix(INTERRUPT_TOOL_CALL_ID_PREFIX)
    if not interrupt_id:
        msg = "Interrupt assistant tool_call ID is invalid."
        raise InvalidResumeRequestError(msg)

    try:
        arguments = _load_json(call.function.arguments)
    except (TypeError, ValueError) as exc:
        msg = "Interrupt assistant tool arguments must be valid JSON."
        raise InvalidResumeRequestError(msg) from exc

    if not isinstance(arguments, dict):
        msg = "Interrupt assistant tool arguments must be a JSON object."
        raise InvalidResumeRequestError(msg)
    if set(arguments) != _INTERRUPT_ARGUMENT_FIELDS:
        msg = (
            "Interrupt assistant tool arguments must contain exactly run_id, "
            "state_token, and payload."
        )
        raise InvalidResumeRequestError(msg)

    run_id = _required_string_argument(arguments, "run_id")
    state_token = _required_string_argument(arguments, "state_token")

    return _InterruptCall(
        interrupt_id=interrupt_id,
        run_id=run_id,
        state_token=state_token,
    )


def _required_string_argument(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        msg = f"Interrupt assistant tool arguments must include {name}."
        raise InvalidResumeRequestError(msg)
    return value


def _parse_tool_results(
    messages: list[ChatCompletionRequestMessage],
    calls: dict[str, _InterruptCall],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in messages:
        tool_call_id = message.tool_call_id
        if not tool_call_id:
            msg = "Interrupt resume tool messages must include tool_call_id."
            raise InvalidResumeRequestError(msg)
        if tool_call_id not in calls:
            msg = "Interrupt resume tool_call_id does not match the assistant request."
            raise InvalidResumeRequestError(msg)
        interrupt_id = calls[tool_call_id].interrupt_id
        if interrupt_id in results:
            msg = "Interrupt resume tool_call_id values must be unique."
            raise InvalidResumeRequestError(msg)

        try:
            payload = _load_json(message.content or "")
        except (TypeError, ValueError) as exc:
            msg = 'Interrupt resume tool content must be JSON like {"resume": "..."}'
            raise InvalidResumeRequestError(msg) from exc

        if not isinstance(payload, dict) or "resume" not in payload:
            msg = 'Interrupt resume tool content must be JSON like {"resume": "..."}'
            raise InvalidResumeRequestError(msg)
        results[interrupt_id] = payload["resume"]

    if set(results) != {call.interrupt_id for call in calls.values()}:
        msg = "Every interrupt tool call must have exactly one tool result."
        raise InvalidResumeRequestError(msg)
    return results


def _load_json(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    msg = f"Unsupported JSON constant: {value}"
    raise ValueError(msg)
