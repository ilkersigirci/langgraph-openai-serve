"""Protocol-neutral encoding for LangGraph interrupt tool calls."""

import json
from dataclasses import dataclass
from typing import Any

from langgraph_openai_serve.graph.interrupt.errors import (
    InvalidInterruptPayloadError,
    InvalidResumeRequestError,
)
from langgraph_openai_serve.graph.interrupt.models import InterruptResume

INTERRUPT_TOOL_NAME = "langgraph_interrupt"
INTERRUPT_TOOL_CALL_ID_PREFIX = "lg_interrupt_"
_INTERRUPT_ARGUMENT_FIELDS = {"run_id", "state_token", "payload"}


@dataclass(frozen=True, slots=True)
class InterruptToolCall:
    """One protocol-decoded interrupt function call."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class InterruptToolOutput:
    """One protocol-decoded interrupt function output."""

    call_id: str
    output: Any


@dataclass(frozen=True, slots=True)
class _ParsedInterruptCall:
    interrupt_id: str
    run_id: str
    state_token: str


def interrupt_tool_call_id(interrupt_id: str) -> str:
    """Format an interrupt's protocol call ID."""
    return f"{INTERRUPT_TOOL_CALL_ID_PREFIX}{interrupt_id}"


def is_interrupt_tool_call_id(call_id: str) -> bool:
    """Return whether a protocol call ID uses the interrupt namespace."""
    return call_id.startswith(INTERRUPT_TOOL_CALL_ID_PREFIX)


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


def parse_interrupt_exchange(
    calls: list[InterruptToolCall],
    outputs: list[InterruptToolOutput],
) -> InterruptResume:
    """Validate a complete interrupt call/output batch."""
    parsed_calls = _parse_interrupt_calls(calls)
    values = _parse_tool_outputs(outputs, parsed_calls)

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


def _parse_interrupt_calls(
    calls: list[InterruptToolCall],
) -> dict[str, _ParsedInterruptCall]:
    parsed: dict[str, _ParsedInterruptCall] = {}
    for call in calls:
        if call.call_id in parsed:
            msg = "Interrupt assistant tool_call IDs must be unique."
            raise InvalidResumeRequestError(msg)
        parsed[call.call_id] = _parse_interrupt_call(call)
    return parsed


def _parse_interrupt_call(call: InterruptToolCall) -> _ParsedInterruptCall:
    if not is_interrupt_tool_call_id(call.call_id):
        msg = "Interrupt assistant tool_call ID is invalid."
        raise InvalidResumeRequestError(msg)

    interrupt_id = call.call_id.removeprefix(INTERRUPT_TOOL_CALL_ID_PREFIX)
    if not interrupt_id:
        msg = "Interrupt assistant tool_call ID is invalid."
        raise InvalidResumeRequestError(msg)

    try:
        arguments = load_json(call.arguments)
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

    return _ParsedInterruptCall(
        interrupt_id=interrupt_id,
        run_id=_required_string_argument(arguments, "run_id"),
        state_token=_required_string_argument(arguments, "state_token"),
    )


def _required_string_argument(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        msg = f"Interrupt assistant tool arguments must include {name}."
        raise InvalidResumeRequestError(msg)
    return value


def _parse_tool_outputs(
    outputs: list[InterruptToolOutput],
    calls: dict[str, _ParsedInterruptCall],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for output in outputs:
        if output.call_id not in calls:
            msg = "Interrupt resume tool_call_id does not match the assistant request."
            raise InvalidResumeRequestError(msg)
        interrupt_id = calls[output.call_id].interrupt_id
        if interrupt_id in results:
            msg = "Interrupt resume tool_call_id values must be unique."
            raise InvalidResumeRequestError(msg)
        if not isinstance(output.output, str):
            msg = 'Interrupt resume tool content must be JSON like {"resume": "..."}'
            raise InvalidResumeRequestError(msg)

        try:
            payload = load_json(output.output)
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


def load_json(value: str) -> Any:
    """Decode strict JSON, rejecting non-standard numeric constants."""
    return json.loads(value, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    msg = f"Unsupported JSON constant: {value}"
    raise ValueError(msg)


__all__ = [
    "INTERRUPT_TOOL_CALL_ID_PREFIX",
    "INTERRUPT_TOOL_NAME",
    "InterruptToolCall",
    "InterruptToolOutput",
    "interrupt_arguments",
    "interrupt_tool_call_id",
    "is_interrupt_tool_call_id",
    "load_json",
    "parse_interrupt_exchange",
]
