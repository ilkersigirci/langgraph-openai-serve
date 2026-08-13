import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from .chainlit_hitl_support import (
    completion,
    install_message_factory,
    interrupt_call,
    ledger_step,
    recording_message,
)


async def test_on_chat_resume_restores_ledger_for_the_next_user_interaction(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    interrupt_calls = [
        interrupt_call("interrupt-1", {"question": "First?"}),
        interrupt_call("interrupt-2", {"question": "Second?"}),
    ]
    step = ledger_step(interrupt_calls)
    restored_writes: list[tuple[str, Mock, dict[str, object] | None]] = []
    restored_message = recording_message(
        metadata=step["metadata"],
        writes=restored_writes,
    )
    factory, created_messages, writes = install_message_factory(
        monkeypatch,
        hitl,
        restored=restored_message,
    )
    create_completion = AsyncMock(return_value=completion("Resumed."))
    ask_for_resume = AsyncMock(side_effect=["approve", "reject"])
    send_ui_message = AsyncMock()
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(hitl, "ask_for_resume", ask_for_resume)
    monkeypatch.setattr(hitl, "send_ui_message", send_ui_message)

    await hitl.on_chat_resume({"steps": [step]})

    factory.from_dict.assert_called_once_with(step)
    create_completion.assert_not_awaited()
    ask_for_resume.assert_not_awaited()
    assert isinstance(
        hitl.cl.user_session.get(hitl.PENDING_LEDGER_SESSION_KEY),
        hitl.PendingInterruptLedger,
    )

    await hitl.handle_message()

    resume_messages = create_completion.await_args.args[0]
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
    create_completion.assert_awaited_once_with(
        resume_messages,
        model_id="lgos-a/hitl",
    )
    assert [write[0] for write in restored_writes] == ["update", "update"]
    assert restored_writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == (
        hitl.PENDING_LEDGER_STATUS
    )
    assert restored_writes[1][2][hitl.INTERRUPT_LEDGER_METADATA_KEY] == {
        "schema_version": hitl.INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": hitl.COMPLETED_LEDGER_STATUS,
    }
    assert step["metadata"][hitl.INTERRUPT_LEDGER_METADATA_KEY] == {
        "schema_version": hitl.INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": hitl.COMPLETED_LEDGER_STATUS,
    }
    assert len(created_messages) == 1
    assert created_messages[0].content == "Resumed."
    assert [write[0] for write in writes] == ["send"]
    send_ui_message.assert_awaited_once_with(
        "Resolve the pending approval before starting another request."
    )


async def test_on_chat_resume_does_not_emit_during_malformed_rehydration(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    step = ledger_step([], raw_ledger={"schema_version": 999})
    factory, _, _ = install_message_factory(monkeypatch, hitl)
    create_completion = AsyncMock()
    ask_for_resume = AsyncMock()
    send_ui_message = AsyncMock()
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(hitl, "ask_for_resume", ask_for_resume)
    monkeypatch.setattr(hitl, "send_ui_message", send_ui_message)

    await hitl.on_chat_resume({"steps": [step]})

    send_ui_message.assert_not_awaited()
    factory.from_dict.assert_not_called()
    create_completion.assert_not_awaited()
    ask_for_resume.assert_not_awaited()


async def test_on_chat_resume_does_not_replay_completed_ledger(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    step = ledger_step(
        [interrupt_call("interrupt-1", {"question": "Approve?"})],
        status="completed",
    )
    factory, _, _ = install_message_factory(monkeypatch, hitl)
    create_completion = AsyncMock()
    ask_for_resume = AsyncMock()
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(hitl, "ask_for_resume", ask_for_resume)

    await hitl.on_chat_resume({"steps": [step]})

    factory.from_dict.assert_not_called()
    create_completion.assert_not_awaited()
    ask_for_resume.assert_not_awaited()
