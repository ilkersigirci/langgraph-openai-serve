import json

import pytest

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage
from langgraph_openai_serve.api.chat.utils.interrupts import (
    InvalidResumeRequestError,
    interrupt_arguments,
    parse_resume_request,
)

RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
STATE_TOKEN = "state-token-1"


def _message(**kwargs) -> ChatCompletionRequestMessage:
    return ChatCompletionRequestMessage.model_validate(kwargs)


def _tool_call(
    interrupt_id: str,
    *,
    run_id: str = RUN_ID,
    state_token: str = STATE_TOKEN,
    arguments: str | None = None,
) -> dict:
    return {
        "id": f"lg_interrupt_{interrupt_id}",
        "type": "function",
        "function": {
            "name": "langgraph_interrupt",
            "arguments": (
                arguments
                if arguments is not None
                else interrupt_arguments(
                    run_id=run_id,
                    state_token=state_token,
                    payload={"question": interrupt_id},
                )
            ),
        },
    }


def _exchange(*interrupt_ids: str) -> list[ChatCompletionRequestMessage]:
    return [
        _message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call(interrupt_id) for interrupt_id in interrupt_ids],
        ),
        *[
            _message(
                role="tool",
                tool_call_id=f"lg_interrupt_{interrupt_id}",
                content=json.dumps({"resume": f"answer:{interrupt_id}"}),
            )
            for interrupt_id in interrupt_ids
        ],
    ]


def test_parse_resume_request_preserves_all_interrupt_ids() -> None:
    resume = parse_resume_request(_exchange("interrupt-1", "interrupt-2"))

    assert resume is not None
    assert resume.run_id == RUN_ID
    assert resume.state_token == STATE_TOKEN
    assert resume.values == {
        "interrupt-1": "answer:interrupt-1",
        "interrupt-2": "answer:interrupt-2",
    }


def test_parse_resume_request_accepts_json_null_by_id() -> None:
    messages = _exchange("interrupt-1")
    messages[-1].content = '{"resume": null}'

    resume = parse_resume_request(messages)

    assert resume is not None
    assert resume.values == {"interrupt-1": None}


def test_parse_resume_request_rejects_nonstandard_json_constants() -> None:
    messages = _exchange("interrupt-1")
    messages[-1].content = '{"resume": NaN}'

    with pytest.raises(InvalidResumeRequestError, match="must be JSON"):
        parse_resume_request(messages)


def test_parse_resume_request_leaves_ordinary_tool_exchange_as_graph_input() -> None:
    messages = [
        _message(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        ),
        _message(role="tool", tool_call_id="call_weather", content="sunny"),
    ]

    assert parse_resume_request(messages) is None


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        pytest.param(
            [
                _message(
                    role="tool",
                    tool_call_id="lg_interrupt_interrupt-1",
                    content='{"resume": "yes"}',
                )
            ],
            "must follow",
            id="missing-assistant-call",
        ),
        pytest.param(
            _exchange("interrupt-1")[:-1],
            None,
            id="no-tool-result-is-not-a-resume",
        ),
        pytest.param(
            [
                _exchange("interrupt-1")[0],
                _message(
                    role="tool",
                    tool_call_id="lg_interrupt_wrong",
                    content='{"resume": "yes"}',
                ),
            ],
            "does not match",
            id="wrong-tool-call-id",
        ),
        pytest.param(
            [
                _exchange("interrupt-1", "interrupt-2")[0],
                _exchange("interrupt-1")[1],
            ],
            "Every interrupt",
            id="incomplete-batch",
        ),
        pytest.param(
            [
                _exchange("interrupt-1")[0],
                _message(
                    role="tool",
                    tool_call_id="lg_interrupt_interrupt-1",
                    content='{"value": "yes"}',
                ),
            ],
            "resume",
            id="missing-resume-value",
        ),
    ],
)
def test_parse_resume_request_rejects_malformed_interrupt_exchange(
    messages: list[ChatCompletionRequestMessage],
    error: str | None,
) -> None:
    if error is None:
        assert parse_resume_request(messages) is None
        return

    with pytest.raises(InvalidResumeRequestError, match=error):
        parse_resume_request(messages)


@pytest.mark.parametrize("payload", [object(), float("nan"), float("inf")])
def test_interrupt_arguments_rejects_non_json_payload(payload: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        interrupt_arguments(
            run_id=RUN_ID,
            state_token=STATE_TOKEN,
            payload=payload,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"run_id": RUN_ID, "state_token": STATE_TOKEN},
        {
            "run_id": RUN_ID,
            "state_token": STATE_TOKEN,
            "payload": {"question": "interrupt-1"},
            "kind": "hitl.interrupt",
        },
    ],
    ids=["missing-field", "extra-field"],
)
def test_parse_resume_request_requires_exact_interrupt_argument_fields(
    arguments: dict,
) -> None:
    messages = _exchange("interrupt-1")
    assert messages[0].tool_calls is not None
    messages[0].tool_calls[0].function.arguments = json.dumps(arguments)

    with pytest.raises(InvalidResumeRequestError, match="contain exactly"):
        parse_resume_request(messages)


def test_parse_resume_request_rejects_duplicate_tool_results() -> None:
    messages = _exchange("interrupt-1")
    messages.append(
        _message(
            role="tool",
            tool_call_id="lg_interrupt_interrupt-1",
            content='{"resume": "second answer"}',
        )
    )

    with pytest.raises(InvalidResumeRequestError, match="must be unique"):
        parse_resume_request(messages)


@pytest.mark.parametrize(
    ("calls", "error"),
    [
        pytest.param(
            [
                _tool_call("interrupt-1"),
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                },
            ],
            "cannot be resumed",
            id="mixed-ordinary-and-interrupt-calls",
        ),
        pytest.param(
            [_tool_call("interrupt-1"), _tool_call("interrupt-1")],
            "tool_call IDs must be unique",
            id="duplicate-assistant-tool-call-id",
        ),
        pytest.param(
            [
                _tool_call("interrupt-1"),
                _tool_call("interrupt-2", run_id="different-run"),
            ],
            "same run",
            id="cross-call-run-id-mismatch",
        ),
        pytest.param(
            [
                _tool_call("interrupt-1"),
                _tool_call("interrupt-2", state_token="different-state"),
            ],
            "interrupt generation",
            id="cross-call-state-token-mismatch",
        ),
        pytest.param(
            [_tool_call("interrupt-1", arguments="{")],
            "must be valid JSON",
            id="malformed-arguments",
        ),
        pytest.param(
            [_tool_call("interrupt-1", arguments="[]")],
            "must be a JSON object",
            id="non-object-arguments",
        ),
        pytest.param(
            [_tool_call("interrupt-1", run_id="")],
            "must include run_id",
            id="empty-run-id",
        ),
        pytest.param(
            [_tool_call("interrupt-1", state_token="")],
            "must include state_token",
            id="empty-state-token",
        ),
    ],
)
def test_parse_resume_request_rejects_invalid_assistant_interrupt_batch(
    calls: list[dict],
    error: str,
) -> None:
    messages = [
        _message(role="assistant", content=None, tool_calls=calls),
        *[
            _message(
                role="tool",
                tool_call_id=call["id"],
                content='{"resume": "answer"}',
            )
            for call in calls
        ],
    ]

    with pytest.raises(InvalidResumeRequestError, match=error):
        parse_resume_request(messages)
