import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from lgos_openwebui.functions import generic
from lgos_openwebui.functions.generic import Pipe

from .openwebui_support import (
    INTERRUPT_PAYLOAD,
    MODEL_ID,
    RUN_ID,
    STATE_TOKEN,
    USER_REQUEST,
    ScriptedChat,
    body,
    collect_response,
    completion,
    interrupt_call,
    interrupt_response,
    model,
    run_interrupt_pipe,
)


@pytest.mark.parametrize(
    ("approved", "decision", "answer_deltas"),
    [
        pytest.param(
            True,
            "approve",
            ("Approved agent action: ", USER_REQUEST),
            id="approve",
        ),
        pytest.param(
            False,
            "reject",
            (f"Rejected agent action: {USER_REQUEST}",),
            id="reject",
        ),
    ],
)
async def test_pipe_resumes_confirmed_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    approved: bool,
    decision: str,
    answer_deltas: tuple[str, ...],
) -> None:
    chat = ScriptedChat(
        ((), interrupt_response()),
        (answer_deltas, completion("".join(answer_deltas))),
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return approved

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, confirm)

    assert chunks == list(answer_deltas)
    assert events == [
        {
            "type": "confirmation",
            "data": {"title": "Approve?", "message": USER_REQUEST},
        }
    ]
    (
        (initial_messages, initial_model_id),
        (resume_messages, resume_model_id),
    ) = chat.calls
    assert initial_messages == [{"role": "user", "content": USER_REQUEST}]
    assert resume_messages[0]["tool_calls"][0]["id"] == ("lg_interrupt_interrupt-1")
    assert json.loads(resume_messages[0]["tool_calls"][0]["function"]["arguments"]) == {
        "run_id": RUN_ID,
        "state_token": STATE_TOKEN,
        "payload": INTERRUPT_PAYLOAD,
    }
    assert resume_messages[1] == {
        "role": "tool",
        "tool_call_id": "lg_interrupt_interrupt-1",
        "content": json.dumps({"resume": decision}),
    }
    assert (initial_model_id, resume_model_id) == (MODEL_ID, MODEL_ID)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param("Approve transfer?", "Approve transfer?", id="string"),
        pytest.param(
            ["transfer", {"amount": 42}],
            '[\n  "transfer",\n  {\n    "amount": 42\n  }\n]',
            id="list",
        ),
        pytest.param(42, "42", id="number"),
        pytest.param(True, "true", id="boolean"),
        pytest.param(None, "null", id="null"),
    ],
)
def test_approval_event_renders_every_json_payload(
    payload: object,
    message: str,
) -> None:
    tool_call = (
        completion(tool_calls=[interrupt_call("interrupt-1", payload)])
        .choices[0]
        .message.tool_calls[0]
    )

    assert generic._interrupt_payload(tool_call) == payload
    assert generic._approval_event(tool_call) == {
        "type": "confirmation",
        "data": {
            "title": "Approve this agent action?",
            "message": message,
        },
    }


def test_approval_event_rejects_a_missing_payload() -> None:
    arguments = {
        "run_id": RUN_ID,
        "state_token": STATE_TOKEN,
    }
    tool_call = (
        completion(
            tool_calls=[
                interrupt_call("interrupt-1", {}, arguments=arguments),
            ]
        )
        .choices[0]
        .message.tool_calls[0]
    )

    assert generic._approval_event(tool_call) is None


async def test_pipe_collects_every_interrupt_and_resumes_once(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    first_call = interrupt_call(
        "interrupt-1",
        {"question": "Approve refund?", "request": USER_REQUEST},
    )
    second_call = interrupt_call(
        "interrupt-2",
        {
            "question": "Approve notification?",
            "request": "Email the customer",
        },
    )
    interrupt_completion = completion(tool_calls=[first_call, second_call])
    chat = ScriptedChat(
        ((), interrupt_completion),
        (("Applied.",), completion("Applied.")),
    )
    events: list[dict[str, Any]] = []
    answers = iter([True, False])

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return next(answers)

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, confirm)

    assert chunks == ["Applied."]
    assert events == [
        {
            "type": "confirmation",
            "data": {"title": "Approve refund?", "message": USER_REQUEST},
        },
        {
            "type": "confirmation",
            "data": {
                "title": "Approve notification?",
                "message": "Email the customer",
            },
        },
    ]
    assert len(chat.calls) == 2
    resume_messages = chat.calls[1][0]
    assert resume_messages[0] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [first_call, second_call],
    }
    assert resume_messages[1:] == [
        {
            "role": "tool",
            "tool_call_id": "lg_interrupt_interrupt-1",
            "content": json.dumps({"resume": "approve"}),
        },
        {
            "role": "tool",
            "tool_call_id": "lg_interrupt_interrupt-2",
            "content": json.dumps({"resume": "reject"}),
        },
    ]


async def test_pipe_keeps_the_ledger_across_interrupt_rounds(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    first_call = interrupt_call("interrupt-1", {"question": "First?"})
    second_call = interrupt_call(
        "interrupt-2",
        {"question": "Second?"},
        state_token="state-token-2",
    )
    chat = ScriptedChat(
        ((), completion(tool_calls=[first_call])),
        ((), completion(tool_calls=[second_call])),
        (("Done.",), completion("Done.")),
    )
    answers = iter([True, False])

    async def confirm(_: dict[str, Any]) -> bool:
        return next(answers)

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, confirm)

    assert chunks == ["Done."]
    assert len(chat.calls) == 3
    first_ledger = chat.calls[1][0]
    second_ledger = chat.calls[2][0]
    assert [message["role"] for message in first_ledger] == ["assistant", "tool"]
    assert [message["role"] for message in second_ledger] == ["assistant", "tool"]
    assert first_ledger[0]["tool_calls"] == [first_call]
    assert second_ledger[0]["tool_calls"] == [second_call]


async def test_pipe_does_not_partially_resume_cancelled_batch(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat(
        (
            (),
            completion(
                tool_calls=[
                    interrupt_call("interrupt-1", {"question": "First?"}),
                    interrupt_call("interrupt-2", {"question": "Second?"}),
                ]
            ),
        )
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool | None:
        events.append(event)
        return None

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, confirm)

    assert chunks == ["Open WebUI approval was cancelled or timed out."]
    assert len(events) == 1
    assert len(chat.calls) == 1


async def test_pipe_reports_host_approval_failure_without_resuming_batch(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat(
        (
            (),
            completion(
                tool_calls=[
                    interrupt_call("interrupt-1", {"question": "First?"}),
                    interrupt_call("interrupt-2", {"question": "Second?"}),
                ]
            ),
        )
    )
    event_call = AsyncMock(
        return_value={
            "error": "Event call timed out. The browser tab may be inactive or closed."
        }
    )
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, event_call)

    assert chunks == [
        "Open WebUI approval failed: Event call timed out. "
        "The browser tab may be inactive or closed."
    ]
    event_call.assert_awaited_once_with(
        {
            "type": "confirmation",
            "data": {
                "title": "First?",
                "message": json.dumps(
                    {"question": "First?"},
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        }
    )
    assert len(chat.calls) == 1


async def test_pipe_reports_empty_host_approval_exception_without_resuming_batch(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat(((), interrupt_response()))
    event_call = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, event_call)

    assert chunks == [
        "Open WebUI approval failed: the confirmation session disconnected or timed out"
    ]
    event_call.assert_awaited_once()
    assert len(chat.calls) == 1


@pytest.mark.parametrize("mixed", [False, True], ids=["ordinary", "mixed"])
async def test_pipe_reports_unsupported_tool_call_batches(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    mixed: bool,
) -> None:
    ordinary_call = {
        "id": "call_other",
        "type": "function",
        "function": {"name": "other_tool", "arguments": "{}"},
    }
    tool_calls = [ordinary_call]
    if mixed:
        tool_calls.insert(0, interrupt_call("interrupt-1", INTERRUPT_PAYLOAD))
    chat = ScriptedChat(((), completion(tool_calls=tool_calls)))
    event_call = AsyncMock(return_value=True)
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, event_call)

    assert chunks == ["Open WebUI received an unsupported tool-call batch."]
    event_call.assert_not_awaited()
    assert len(chat.calls) == 1


async def test_pipe_does_not_resume_a_malformed_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat(
        ((), interrupt_response([])),
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return True

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await run_interrupt_pipe(configured_pipe, confirm)

    assert chunks == ["Open WebUI received an unsupported interrupt payload."]
    assert events == []
    assert len(chat.calls) == 1


async def test_pipe_passes_runtime_settings_to_initial_and_resume_requests(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat(
        ((), interrupt_response()),
        (("Approved.",), completion("Approved.")),
    )
    runtime_metadata = {
        "langgraph_runtime_settings": '{"use_history":true}',
    }
    settings_metadata = Mock(return_value=runtime_metadata)
    expected_request_metadata = {
        **runtime_metadata,
        "session_id": "chat-123",
    }

    async def confirm(_: dict[str, Any]) -> bool:
        return True

    monkeypatch.setattr(generic, "_chat", chat)
    monkeypatch.setattr(generic, "_runtime_settings_metadata", settings_metadata)

    chunks = await collect_response(
        configured_pipe.pipe(
            body=body(USER_REQUEST),
            __event_call__=confirm,
            __metadata__={
                "chat_id": "chat-123",
                "chat_variables": {"use_history": True},
            },
        )
    )

    assert chunks == ["Approved."]
    assert chat.request_metadata_calls == [
        expected_request_metadata,
        expected_request_metadata,
    ]
    settings_metadata.assert_called_once_with(
        model=model(),
        metadata={
            "chat_id": "chat-123",
            "chat_variables": {"use_history": True},
        },
    )
