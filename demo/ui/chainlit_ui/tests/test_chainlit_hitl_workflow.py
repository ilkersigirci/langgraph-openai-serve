import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from .chainlit_hitl_support import (
    RUN_ID,
    STATE_TOKEN,
    completion,
    install_message_factory,
    interrupt_call,
    recording_message,
)


async def test_handle_message_resumes_all_interrupts_once(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    payloads = [
        {"question": "Approve refund?", "request": "ORDER-123"},
        {"question": "Approve notification?", "request": "Email customer"},
    ]
    interrupt_calls = [
        interrupt_call("interrupt-1", payloads[0]),
        interrupt_call("interrupt-2", payloads[1]),
    ]
    create_completion = AsyncMock(
        side_effect=[
            completion(tool_calls=interrupt_calls),
            completion("Both decisions applied."),
        ]
    )
    _, created_messages, writes = install_message_factory(monkeypatch, hitl)
    decisions = iter(["approve", "reject"])

    async def decide(_: object, ledger_message: object) -> str:
        assert writes[0][0] == "send"
        assert writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == (
            hitl.PENDING_LEDGER_STATUS
        )
        assert ledger_message is created_messages[0]
        return next(decisions)

    ask_for_resume = AsyncMock(side_effect=decide)
    initial_messages = [{"role": "user", "content": "Process both actions."}]
    monkeypatch.setattr(
        hitl,
        "text_only_chat_messages",
        Mock(return_value=list(initial_messages)),
    )
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(hitl, "ask_for_resume", ask_for_resume)
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))

    await hitl.handle_message()

    assert create_completion.await_count == 2
    resume_messages = create_completion.await_args_list[1].args[0]
    assert resume_messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": interrupt_calls,
        },
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
    assert json.loads(interrupt_calls[0]["function"]["arguments"]) == {
        "run_id": RUN_ID,
        "state_token": STATE_TOKEN,
        "payload": payloads[0],
    }
    create_completion.assert_any_await(
        initial_messages,
        model_id="lgos-a/hitl",
    )
    create_completion.assert_any_await(
        resume_messages,
        model_id="lgos-a/hitl",
    )
    assert [awaited.args[0].id for awaited in ask_for_resume.await_args_list] == [
        "lg_interrupt_interrupt-1",
        "lg_interrupt_interrupt-2",
    ]
    ledger_message, output_message = created_messages
    assert ledger_message.content == "Approve refund?\n\nRequest: ORDER-123"
    assert [write[0] for write in writes] == ["send", "update", "send"]
    pending_metadata = writes[0][2]
    assert pending_metadata["lgos_chainlit.exclude_from_model_context"] is True
    assert pending_metadata[hitl.INTERRUPT_LEDGER_METADATA_KEY] == {
        "schema_version": hitl.INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": hitl.PENDING_LEDGER_STATUS,
        "model_id": "lgos-a/hitl",
        "assistant_message": {
            "role": "assistant",
            "content": None,
            "tool_calls": interrupt_calls,
        },
    }
    assert writes[1][1] is ledger_message
    assert writes[1][2][hitl.INTERRUPT_LEDGER_METADATA_KEY] == {
        "schema_version": hitl.INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": hitl.COMPLETED_LEDGER_STATUS,
    }
    output_message.send.assert_awaited_once_with()


async def test_handle_message_keeps_the_ledger_across_interrupt_rounds(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    first_call = interrupt_call("interrupt-1", {"question": "First?"})
    second_call = interrupt_call(
        "interrupt-2",
        {"question": "Second?"},
        state_token="state-token-2",
    )
    create_completion = AsyncMock(
        side_effect=[
            completion(tool_calls=[first_call]),
            completion(tool_calls=[second_call]),
            completion("Done."),
        ]
    )
    _, created_messages, writes = install_message_factory(monkeypatch, hitl)
    monkeypatch.setattr(
        hitl,
        "text_only_chat_messages",
        Mock(return_value=[{"role": "user", "content": "Begin."}]),
    )
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(
        hitl,
        "ask_for_resume",
        AsyncMock(side_effect=["approve", "reject"]),
    )
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))

    await hitl.handle_message()

    assert create_completion.await_count == 3
    first_resume = create_completion.await_args_list[1].args[0]
    second_resume = create_completion.await_args_list[2].args[0]
    assert [message["role"] for message in first_resume] == ["assistant", "tool"]
    assert [message["role"] for message in second_resume] == ["assistant", "tool"]
    assert first_resume[0]["tool_calls"] == [first_call]
    assert second_resume[0]["tool_calls"] == [second_call]
    ledger_message, output_message = created_messages
    assert [write[0] for write in writes] == ["send", "update", "update", "send"]
    assert all(write[1] is ledger_message for write in writes[:3])
    assert writes[1][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["assistant_message"][
        "tool_calls"
    ] == [second_call]
    assert writes[1][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == (
        hitl.PENDING_LEDGER_STATUS
    )
    assert writes[2][2][hitl.INTERRUPT_LEDGER_METADATA_KEY] == {
        "schema_version": hitl.INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": hitl.COMPLETED_LEDGER_STATUS,
    }
    output_message.send.assert_awaited_once_with()


async def test_handle_message_does_not_partially_resume_cancelled_batch(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    create_completion = AsyncMock(
        return_value=completion(
            tool_calls=[
                interrupt_call("interrupt-1", {"question": "First?"}),
                interrupt_call("interrupt-2", {"question": "Second?"}),
            ]
        )
    )
    ask_for_resume = AsyncMock(return_value=None)
    _, created_messages, writes = install_message_factory(monkeypatch, hitl)
    monkeypatch.setattr(hitl, "text_only_chat_messages", Mock(return_value=[]))
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(hitl, "ask_for_resume", ask_for_resume)
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))

    await hitl.handle_message()

    assert ask_for_resume.await_count == 1
    create_completion.assert_awaited_once_with([], model_id="lgos-a/hitl")
    assert len(created_messages) == 1
    assert [write[0] for write in writes] == ["send"]
    assert writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == (
        hitl.PENDING_LEDGER_STATUS
    )


async def test_next_message_reprompts_live_pending_ledger_without_new_run(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    interrupt = interrupt_call(
        "interrupt-1",
        {"question": "Approve?"},
    )
    create_completion = AsyncMock(
        side_effect=[
            completion(tool_calls=[interrupt]),
            completion("Approved."),
        ]
    )
    session: dict[str, object] = {}
    _, _, writes = install_message_factory(
        monkeypatch,
        hitl,
        session=session,
    )
    text_only_chat_messages = Mock(return_value=[{"role": "user", "content": "Start."}])
    monkeypatch.setattr(hitl, "text_only_chat_messages", text_only_chat_messages)
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(
        hitl,
        "ask_for_resume",
        AsyncMock(side_effect=[None, "approve"]),
    )
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))
    send_ui_message = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", send_ui_message)
    trigger_writes: list[tuple[str, Mock, dict[str, object] | None]] = []
    trigger_message = recording_message(
        content="Continue.",
        writes=trigger_writes,
    )

    await hitl.handle_message()
    await hitl.handle_message(trigger_message)

    assert create_completion.await_count == 2
    assert create_completion.await_args_list[0].args[0] == [
        {"role": "user", "content": "Start."}
    ]
    assert [
        message["role"] for message in create_completion.await_args_list[1].args[0]
    ] == [
        "assistant",
        "tool",
    ]
    assert text_only_chat_messages.call_count == 1
    send_ui_message.assert_awaited_once_with(
        "Resolve the pending interrupt before starting another request."
    )
    assert session[hitl.PENDING_LEDGER_SESSION_KEY] is None
    assert [write[0] for write in writes] == ["send", "update", "update", "send"]
    assert [write[0] for write in trigger_writes] == ["update"]
    assert trigger_writes[0][2] == {"lgos_chainlit.exclude_from_model_context": True}
