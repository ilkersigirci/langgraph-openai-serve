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

INTERRUPT_TOOL_NAME = "langgraph_interrupt"
_INTERRUPT_CALL_PREFIX = "call_lg_"
_INTERRUPT_RESPONSE_PREFIX = "resp_lg_"
_STATE_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RESPONSE_ID_PATTERN = re.compile(
    rf"^{_INTERRUPT_RESPONSE_PREFIX}(?P<run>[0-9a-f]{{32}})_[0-9a-f]{{32}}$"
)


def interrupt_response_id(run_id: str) -> str:
    """Create a unique Response ID that carries its interrupt run identity."""
    return f"{_INTERRUPT_RESPONSE_PREFIX}{uuid.UUID(run_id).hex}_{uuid.uuid4().hex}"


def interrupt_tool_call_id(
    interrupt_id: str, state_token: str, *, response_id: str
) -> str:
    """Bind one interrupt to its Response and durable checkpoint generation."""
    if not interrupt_id:
        msg = "LangGraph interrupt IDs must be non-empty strings."
        raise ValueError(msg)
    if _STATE_TOKEN_PATTERN.fullmatch(state_token) is None:
        msg = "LangGraph interrupt state tokens must be SHA-256 hex digests."
        raise ValueError(msg)
    response_nonce = response_id.rsplit("_", 1)[-1]
    return f"{_INTERRUPT_CALL_PREFIX}{state_token}_{response_nonce}_{interrupt_id}"


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

    run_id = _parse_interrupt_response_id(previous_response_id)
    state_token: str | None = None
    values: dict[str, str] = {}
    for item in input_value:
        if not isinstance(item, ResponseFunctionCallOutputInput):
            msg = (
                "Interrupt resumes require only function_call_output input items for "
                "the previous Response."
            )
            raise InvalidResumeRequestError(msg)
        output_token, interrupt_id = _parse_interrupt_tool_call_id(
            item.call_id, previous_response_id
        )
        if state_token is None:
            state_token = output_token
        elif output_token != state_token:
            msg = "Interrupt outputs must belong to one checkpoint generation."
            raise InvalidResumeRequestError(msg)
        if interrupt_id in values:
            msg = "Interrupt function_call_output call_id values must be unique."
            raise InvalidResumeRequestError(msg)
        values[interrupt_id] = item.output

    if state_token is None:  # Response input lists are non-empty by schema.
        msg = "Interrupt resumes require at least one function_call_output item."
        raise InvalidResumeRequestError(msg)
    return InterruptResume(
        run_id=run_id,
        state_token=state_token,
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
            and item.name == INTERRUPT_TOOL_NAME
        )
        or (
            isinstance(item, ResponseFunctionCallOutputInput)
            and item.call_id.startswith(_INTERRUPT_CALL_PREFIX)
        )
        for item in input_value
    ):
        msg = "Interrupt resumes require previous_response_id."
        raise InvalidResumeRequestError(msg)


def _parse_interrupt_response_id(response_id: str) -> str:
    match = _RESPONSE_ID_PATTERN.fullmatch(response_id)
    if match is None:
        msg = "previous_response_id is not an LGOS interrupt Response ID."
        raise InvalidResumeRequestError(msg, param="previous_response_id")
    return str(uuid.UUID(hex=match.group("run")))


def _parse_interrupt_tool_call_id(
    call_id: str, previous_response_id: str
) -> tuple[str, str]:
    if not call_id.startswith(_INTERRUPT_CALL_PREFIX):
        msg = "Interrupt function_call_output call_id is invalid."
        raise InvalidResumeRequestError(msg)
    state_token, _, response_call = call_id.removeprefix(
        _INTERRUPT_CALL_PREFIX
    ).partition("_")
    response_nonce, separator, interrupt_id = response_call.partition("_")
    if (
        not separator
        or not interrupt_id
        or _STATE_TOKEN_PATTERN.fullmatch(state_token) is None
    ):
        msg = "Interrupt function_call_output call_id is invalid."
        raise InvalidResumeRequestError(msg)
    if response_nonce != previous_response_id.rsplit("_", 1)[-1]:
        msg = "Interrupt outputs do not belong to previous_response_id."
        raise InvalidResumeRequestError(msg, param="previous_response_id")
    return state_token, interrupt_id


__all__ = [
    "INTERRUPT_TOOL_NAME",
    "interrupt_response_id",
    "interrupt_tool_call_id",
    "parse_responses_resume",
]
