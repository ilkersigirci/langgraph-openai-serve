"""Unit coverage for the Responses interrupt continuation codec."""

import json

import pytest

from langgraph_openai_serve.api.responses.interrupts import (
    INTERRUPT_TOOL_NAME,
    interrupt_arguments,
    interrupt_response_id,
    interrupt_tool_call_id,
    parse_responses_resume,
)
from langgraph_openai_serve.api.responses.schemas import (
    ResponseFunctionCallInput,
    ResponseFunctionCallOutputInput,
    ResponseInputItem,
)
from langgraph_openai_serve.graph.interrupt.errors import InvalidResumeRequestError

RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
STATE_TOKEN = "a" * 64


def _output(
    interrupt_id: str,
    output: str,
    *,
    state_token: str = STATE_TOKEN,
) -> ResponseFunctionCallOutputInput:
    return ResponseFunctionCallOutputInput(
        call_id=interrupt_tool_call_id(interrupt_id, state_token),
        output=output,
    )


def test_parse_responses_resume_returns_none_without_previous_response() -> None:
    assert parse_responses_resume("hello") is None


def test_parse_responses_resume_preserves_complete_string_output_batch() -> None:
    outputs: list[ResponseInputItem] = [
        _output("interrupt-1", "approved"),
        _output("interrupt-2", "null"),
    ]

    resume = parse_responses_resume(
        outputs,
        previous_response_id=interrupt_response_id(RUN_ID),
    )

    assert resume is not None
    assert resume.run_id == RUN_ID
    assert resume.state_token == STATE_TOKEN
    assert resume.values == {
        "interrupt-1": "approved",
        "interrupt-2": "null",
    }


@pytest.mark.parametrize(
    "input_value",
    [
        pytest.param("answer", id="string-input"),
        pytest.param(
            [
                ResponseFunctionCallInput(
                    call_id="call_weather",
                    name="weather",
                    arguments="{}",
                )
            ],
            id="function-call-input",
        ),
        pytest.param(
            [
                _output("interrupt-1", "yes"),
                ResponseFunctionCallInput(
                    call_id="call_weather",
                    name="weather",
                    arguments="{}",
                ),
            ],
            id="mixed-input",
        ),
    ],
)
def test_previous_response_id_requires_only_function_outputs(
    input_value: str | list[ResponseInputItem],
) -> None:
    with pytest.raises(InvalidResumeRequestError, match="only function_call_output"):
        parse_responses_resume(
            input_value,
            previous_response_id=interrupt_response_id(RUN_ID),
        )


def test_interrupt_items_require_previous_response_id() -> None:
    call_id = interrupt_tool_call_id("interrupt-1", STATE_TOKEN)
    items: list[ResponseInputItem] = [
        ResponseFunctionCallInput(
            call_id=call_id,
            name=INTERRUPT_TOOL_NAME,
            arguments='{"question":"Approve?"}',
        ),
        ResponseFunctionCallOutputInput(call_id=call_id, output="yes"),
    ]

    with pytest.raises(InvalidResumeRequestError, match="previous_response_id"):
        parse_responses_resume(items)


@pytest.mark.parametrize(
    "previous_response_id",
    ["resp_prior", f"resp_lg_{RUN_ID.replace('-', '')}", ""],
)
def test_parse_responses_resume_rejects_invalid_previous_response_id(
    previous_response_id: str,
) -> None:
    with pytest.raises(InvalidResumeRequestError, match="interrupt Response ID"):
        parse_responses_resume(
            [_output("interrupt-1", "yes")],
            previous_response_id=previous_response_id,
        )


@pytest.mark.parametrize(
    "call_id",
    [
        "call_weather",
        "call_lg_missing-token",
        f"call_lg_{STATE_TOKEN}_",
        f"call_lg_{'z' * 64}_interrupt-1",
    ],
)
def test_parse_responses_resume_rejects_invalid_call_id(call_id: str) -> None:
    output = ResponseFunctionCallOutputInput(call_id=call_id, output="yes")

    with pytest.raises(InvalidResumeRequestError, match="call_id is invalid"):
        parse_responses_resume(
            [output],
            previous_response_id=interrupt_response_id(RUN_ID),
        )


def test_parse_responses_resume_rejects_mixed_generations() -> None:
    with pytest.raises(InvalidResumeRequestError, match="one checkpoint generation"):
        parse_responses_resume(
            [
                _output("interrupt-1", "yes"),
                _output("interrupt-2", "no", state_token="b" * 64),
            ],
            previous_response_id=interrupt_response_id(RUN_ID),
        )


def test_parse_responses_resume_rejects_duplicate_outputs() -> None:
    with pytest.raises(InvalidResumeRequestError, match="must be unique"):
        parse_responses_resume(
            [
                _output("interrupt-1", "yes"),
                _output("interrupt-1", "no"),
            ],
            previous_response_id=interrupt_response_id(RUN_ID),
        )


def test_interrupt_response_ids_are_unique_and_keep_run_identity() -> None:
    first = interrupt_response_id(RUN_ID)
    second = interrupt_response_id(RUN_ID)

    assert first != second
    assert first.startswith(f"resp_lg_{RUN_ID.replace('-', '')}_")
    resume = parse_responses_resume(
        [_output("interrupt-1", "yes")],
        previous_response_id=first,
    )
    assert resume is not None
    assert resume.run_id == RUN_ID


def test_interrupt_arguments_are_compact_json() -> None:
    assert interrupt_arguments({"question": "Approve?"}) == json.dumps(
        {"question": "Approve?"}, separators=(",", ":")
    )


def test_interrupt_arguments_reject_non_json_values() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        interrupt_arguments({"value": float("nan")})
