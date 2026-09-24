"""OpenAI Responses encoding for LangGraph interrupt continuations."""

import re
import uuid

from langgraph_openai_serve.api.responses.schemas import (
    ResponseFunctionCallInput,
    ResponseFunctionCallOutputInput,
    ResponseInputItem,
)
from langgraph_openai_serve.graph.interrupt.errors import InvalidResumeRequestError
from langgraph_openai_serve.graph.interrupt.models import InterruptResume
from langgraph_openai_serve.protocol import INTERRUPT_TOOL_NAME as _INTERRUPT_TOOL_NAME

_INTERRUPT_CALL_PREFIX = "call_lg_"
_INTERRUPT_RESPONSE_PREFIX = "resp_lg_"
_RESPONSE_ID_PATTERN = re.compile(
    rf"^{_INTERRUPT_RESPONSE_PREFIX}(?P<run>[0-9a-f]{{32}})_[0-9a-f]{{32}}$"
)


def interrupt_response_id(run_id: str) -> str:
    """Create a unique Response ID that carries its interrupt run identity."""
    return f"{_INTERRUPT_RESPONSE_PREFIX}{uuid.UUID(run_id).hex}_{uuid.uuid4().hex}"


def interrupt_tool_call_id(interrupt_id: str) -> str:
    """Expose one native LangGraph interrupt ID as an OpenAI call ID."""
    if not interrupt_id:
        msg = "LangGraph interrupt IDs must be non-empty strings."
        raise ValueError(msg)
    return f"{_INTERRUPT_CALL_PREFIX}{interrupt_id}"


def parse_responses_resume(
    input_value: str | list[ResponseInputItem],
    *,
    previous_response_id: str | None = None,
) -> InterruptResume | None:
    """Parse the sole supported interrupt continuation form."""
    if previous_response_id is None:
        _reject_interrupt_items_without_response_id(input_value)
        return None
    if isinstance(input_value, str):
        msg = (
            "Interrupt resumes require only function_call_output input items for "
            "the previous Response."
        )
        raise InvalidResumeRequestError(msg)

    run_id = interrupt_run_id(previous_response_id)
    values: dict[str, str] = {}
    for item in input_value:
        if not isinstance(item, ResponseFunctionCallOutputInput):
            msg = (
                "Interrupt resumes require only function_call_output input items for "
                "the previous Response."
            )
            raise InvalidResumeRequestError(msg)
        interrupt_id = _parse_interrupt_tool_call_id(item.call_id)
        if interrupt_id in values:
            msg = "Interrupt function_call_output call_id values must be unique."
            raise InvalidResumeRequestError(msg)
        values[interrupt_id] = item.output

    if not values:  # Response input lists are non-empty by schema.
        msg = "Interrupt resumes require at least one function_call_output item."
        raise InvalidResumeRequestError(msg)
    return InterruptResume(
        run_id=run_id,
        values=values,
    )


def _reject_interrupt_items_without_response_id(
    input_value: str | list[ResponseInputItem],
) -> None:
    if isinstance(input_value, str):
        return
    if any(
        (
            isinstance(item, ResponseFunctionCallInput)
            and item.name == _INTERRUPT_TOOL_NAME
        )
        or (
            isinstance(item, ResponseFunctionCallOutputInput)
            and item.call_id.startswith(_INTERRUPT_CALL_PREFIX)
        )
        for item in input_value
    ):
        msg = "Interrupt resumes require previous_response_id."
        raise InvalidResumeRequestError(msg)


def interrupt_run_id(response_id: str) -> str:
    """Return the run ID carried by an interrupt-style Response ID."""
    match = _RESPONSE_ID_PATTERN.fullmatch(response_id)
    if match is None or uuid.UUID(hex=match.group("run")).int == 0:
        msg = "previous_response_id is not an LGOS interrupt Response ID."
        raise InvalidResumeRequestError(msg, param="previous_response_id")
    return str(uuid.UUID(hex=match.group("run")))


def _parse_interrupt_tool_call_id(call_id: str) -> str:
    interrupt_id = call_id.removeprefix(_INTERRUPT_CALL_PREFIX)
    if interrupt_id == call_id or not interrupt_id:
        msg = "Interrupt function_call_output call_id is invalid."
        raise InvalidResumeRequestError(msg)
    return interrupt_id


__all__ = [
    "interrupt_response_id",
    "interrupt_run_id",
    "interrupt_tool_call_id",
    "parse_responses_resume",
]
