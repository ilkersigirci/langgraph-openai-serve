"""Responses-native Chainlit interrupt workflow and persistence tests."""

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
RESPONSE_ID = f"resp_lg_{RUN_ID.replace('-', '')}_{'b' * 32}"


def response(*output: object, response_id: str = RESPONSE_ID) -> Response:
    return Response.model_construct(
        id=response_id,
        status="completed",
        output=list(output),
    )


def final_response(content: str) -> Response:
    return response(
        ResponseOutputMessage(
            id="msg_final",
            content=[
                ResponseOutputText(
                    annotations=[],
                    logprobs=[],
                    text=content,
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
            phase="final_answer",
        )
    )


def interrupt_call(
    suffix: str,
    payload: object,
    *,
    state_token: str = "a" * 64,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        call_id=f"call_lg_{state_token}_{suffix}",
        name="langgraph_interrupt",
        arguments=json.dumps(payload, separators=(",", ":")),
        status="completed",
        type="function_call",
    )


def recording_message(
    writes: list[tuple[str, Mock, dict[str, object] | None]],
    *,
    content: str = "",
    metadata: dict[str, object] | None = None,
) -> Mock:
    message = Mock(content=content, metadata=metadata)

    async def send() -> Mock:
        writes.append(("send", message, deepcopy(message.metadata)))
        return message

    async def update() -> bool:
        writes.append(("update", message, deepcopy(message.metadata)))
        return True

    message.send = AsyncMock(side_effect=send)
    message.update = AsyncMock(side_effect=update)
    return message


def install_chainlit(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
    *,
    restored: Mock | None = None,
) -> tuple[list[Mock], list[tuple[str, Mock, dict[str, object] | None]]]:
    created: list[Mock] = []
    writes: list[tuple[str, Mock, dict[str, object] | None]] = []

    def create_message(content: str = "", **_: object) -> Mock:
        message = recording_message(writes, content=content)
        created.append(message)
        return message

    factory = Mock(side_effect=create_message)
    factory.from_dict = Mock(return_value=restored)
    monkeypatch.setattr(hitl.cl, "Message", factory)
    values: dict[str, object] = {}
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: values.get(key, default),
            set=values.__setitem__,
        ),
    )
    return created, writes


async def test_handle_message_uploads_files_and_uses_responses(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    input_items = [{"role": "user", "content": "Review this."}]
    with_file = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Review this."},
                {"type": "input_file", "file_id": "file-123"},
            ],
        }
    ]
    upload = AsyncMock(return_value=with_file)
    create = AsyncMock(return_value=final_response("Done."))
    created, _ = install_chainlit(monkeypatch, hitl)
    monkeypatch.setattr(hitl, "text_only_chat_messages", Mock(return_value=input_items))
    monkeypatch.setattr(hitl, "with_response_file_parts", upload)
    monkeypatch.setattr(hitl, "create_response", create)
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))
    trigger = Mock()

    await hitl.handle_message(trigger)

    upload.assert_awaited_once_with(input_items, trigger)
    create.assert_awaited_once_with(with_file, model_id="lgos-a/hitl")
    assert created[-1].content == "Done."


async def test_interrupt_batch_uses_previous_response_id_with_small_outputs(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    calls = [
        interrupt_call("one", {"question": "Approve refund?"}),
        interrupt_call("two", {"question": "Notify customer?"}),
    ]
    create = AsyncMock(side_effect=[response(*calls), final_response("Applied.")])
    created, writes = install_chainlit(monkeypatch, hitl)
    monkeypatch.setattr(
        hitl,
        "text_only_chat_messages",
        Mock(return_value=[{"role": "user", "content": "Process both."}]),
    )
    monkeypatch.setattr(hitl, "create_response", create)
    monkeypatch.setattr(
        hitl, "ask_for_resume", AsyncMock(side_effect=["approve", "reject"])
    )
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))

    await hitl.handle_message()

    continuation = create.await_args_list[1].args[0]
    assert [item["type"] for item in continuation] == [
        "function_call_output",
        "function_call_output",
    ]
    assert [item["call_id"] for item in continuation] == [
        calls[0].call_id,
        calls[1].call_id,
    ]
    assert continuation[0]["output"] == "approve"
    assert create.await_args_list[1].kwargs["previous_response_id"] == RESPONSE_ID
    ledger = writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]
    assert ledger["response_id"] == RESPONSE_ID
    assert ledger["function_calls"] == [
        call.model_dump(mode="json", exclude_none=True) for call in calls
    ]
    assert writes[-2][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == "completed"
    assert created[-1].content == "Applied."


def test_pending_ledger_round_trips_response_function_calls(hitl: Any) -> None:
    calls = [interrupt_call("one", {"question": "Approve?"})]
    raw = {
        "schema_version": hitl.INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": hitl.PENDING_LEDGER_STATUS,
        "model_id": "lgos-a/hitl",
        "response_id": RESPONSE_ID,
        "function_calls": [
            call.model_dump(mode="json", exclude_none=True) for call in calls
        ],
    }

    model_id, response_id, restored = hitl.parse_interrupt_ledger_metadata(raw)

    assert model_id == "lgos-a/hitl"
    assert response_id == RESPONSE_ID
    assert restored == calls


async def test_resumed_ledger_uses_responses_continuation(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    call = interrupt_call("one", {"question": "Approve?"})
    writes: list[tuple[str, Mock, dict[str, object] | None]] = []
    ledger_message = recording_message(writes)
    create = AsyncMock(return_value=final_response("Resumed."))
    created, _ = install_chainlit(monkeypatch, hitl)
    monkeypatch.setattr(hitl, "create_response", create)
    monkeypatch.setattr(hitl, "ask_for_resume", AsyncMock(return_value="approve"))

    await hitl.resolve_interrupts(
        hitl.PendingInterruptLedger(
            message=ledger_message,
            response_id=RESPONSE_ID,
            model_id="lgos-a/hitl",
            calls=(call,),
        )
    )

    continuation = create.await_args.args[0]
    assert continuation == [
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": "approve",
        }
    ]
    assert create.await_args.kwargs["previous_response_id"] == RESPONSE_ID
    assert created[-1].content == "Resumed."


async def test_cancelled_batch_remains_pending_without_partial_resume(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    calls = [
        interrupt_call("one", {"question": "First?"}),
        interrupt_call("two", {"question": "Second?"}),
    ]
    create = AsyncMock(return_value=response(*calls))
    _, writes = install_chainlit(monkeypatch, hitl)
    monkeypatch.setattr(hitl, "text_only_chat_messages", Mock(return_value=[]))
    monkeypatch.setattr(hitl, "create_response", create)
    monkeypatch.setattr(hitl, "ask_for_resume", AsyncMock(return_value=None))
    monkeypatch.setattr(hitl, "selected_model_id", Mock(return_value="lgos-a/hitl"))

    await hitl.handle_message()

    create.assert_awaited_once_with([], model_id="lgos-a/hitl")
    assert writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == "pending"


def test_interrupt_payload_and_prompt_are_decoded_from_sdk_call(hitl: Any) -> None:
    payload = {
        "question": "Approve refund?",
        "request": "ORDER-123",
        "choices": ["approve", "reject"],
    }
    call = interrupt_call("one", payload)

    assert hitl.interrupt_payload(call) == payload
    assert hitl.pending_interrupt_prompt([call]) == (
        "Approve refund?\n\nRequest: ORDER-123"
    )


def test_invalid_persisted_call_is_rejected(hitl: Any) -> None:
    with pytest.raises(hitl.InvalidInterruptLedgerError):
        hitl.parse_interrupt_ledger_metadata(
            {
                "schema_version": 1,
                "status": "pending",
                "model_id": "lgos-a/hitl",
                "function_calls": [{"type": "function_call"}],
            }
        )


async def test_failed_response_keeps_pending_interrupt_ledger(monkeypatch, hitl):
    calls = [interrupt_call("one", {"question": "Approve?"})]
    created, writes = install_chainlit(monkeypatch, hitl)
    failed = response()
    failed.status = "failed"
    failed.error = SimpleNamespace(message="Resume failed")
    monkeypatch.setattr(
        hitl.openai_client.responses, "create", AsyncMock(return_value=failed)
    )
    monkeypatch.setattr(hitl, "model_request", lambda _: {"model": "hitl"})
    monkeypatch.setattr(hitl, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(hitl, "session_metadata", dict)
    monkeypatch.setattr(hitl, "ask_for_resume", AsyncMock(return_value="approve"))

    with pytest.raises(RuntimeError, match="Resume failed"):
        pending = await hitl.publish_response(
            response(*calls),
            model_id="hitl",
        )
        await hitl.resolve_interrupts(pending)

    assert writes[-1][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == "pending"
    assert len(created) == 1


@pytest.mark.parametrize("payload", [[], None, "approve"])
async def test_invalid_interrupt_payload_is_persisted_before_prompting(
    monkeypatch, hitl, payload
):
    call = interrupt_call("one", payload)
    _, writes = install_chainlit(monkeypatch, hitl)
    notify = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", notify)
    monkeypatch.setattr(hitl, "create_response", create)

    pending = await hitl.publish_response(response(call), model_id="hitl")
    await hitl.resolve_interrupts(pending)

    saved = writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]
    assert saved["response_id"] == RESPONSE_ID
    assert saved["function_calls"][0]["arguments"] == call.arguments
    assert saved["status"] == "pending"
    notify.assert_awaited_once_with("Received an unsupported interrupt payload.")
    create.assert_not_awaited()


async def test_sequential_interrupts_replace_the_ledger_before_prompting(
    monkeypatch, hitl
):
    first = interrupt_call("one", {"question": "First?"})
    second = interrupt_call("two", {"question": "Second?"})
    second_response_id = "resp_second"
    _, writes = install_chainlit(monkeypatch, hitl)
    create = AsyncMock(
        side_effect=[
            response(second, response_id=second_response_id),
            final_response("Both approved."),
        ]
    )
    monkeypatch.setattr(hitl, "create_response", create)
    seen = []

    async def approve(call, message):
        saved = deepcopy(message.metadata[hitl.INTERRUPT_LEDGER_METADATA_KEY])
        seen.append(saved)
        assert saved["function_calls"][0]["call_id"] == call.call_id
        return "approve"

    monkeypatch.setattr(hitl, "ask_for_resume", approve)
    pending = await hitl.publish_response(response(first), model_id="hitl")
    await hitl.resolve_interrupts(pending)

    assert [saved["response_id"] for saved in seen] == [RESPONSE_ID, second_response_id]
    assert [call.kwargs["previous_response_id"] for call in create.await_args_list] == [
        RESPONSE_ID,
        second_response_id,
    ]
    assert writes[-2][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == "completed"
