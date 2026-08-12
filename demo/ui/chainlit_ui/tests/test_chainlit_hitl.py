import importlib
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types import Model
from openai.types.chat import ChatCompletion

from lgos_chainlit.lgos_protocol import (
    GraphFeature,
    model_description,
    model_supports,
)

RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
STATE_TOKEN = "state-token-1"


@pytest.fixture
def hitl() -> Any:
    return importlib.import_module("lgos_chainlit.hitl")


def _recording_message(
    *,
    content: str = "",
    metadata: dict[str, object] | None = None,
    writes: list[tuple[str, Mock, dict[str, object] | None]],
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


def _install_message_factory(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
    *,
    restored: Mock | None = None,
) -> tuple[
    Mock,
    list[Mock],
    list[tuple[str, Mock, dict[str, object] | None]],
]:
    created: list[Mock] = []
    writes: list[tuple[str, Mock, dict[str, object] | None]] = []

    def create_message(
        content: str = "",
        metadata: dict[str, object] | None = None,
        **_: object,
    ) -> Mock:
        message = _recording_message(
            content=content,
            metadata=metadata,
            writes=writes,
        )
        created.append(message)
        return message

    factory = Mock(side_effect=create_message)
    factory.from_dict = Mock(return_value=restored)
    monkeypatch.setattr(hitl.cl, "Message", factory)
    session: dict[str, object] = {}
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: session.get(key, default),
            set=lambda key, value: session.__setitem__(key, value),
        ),
    )
    return factory, created, writes


def _interrupt_call(
    interrupt_id: str,
    payload: object,
    *,
    arguments: object | None = None,
    state_token: str = STATE_TOKEN,
) -> dict[str, object]:
    arguments = (
        {
            "run_id": RUN_ID,
            "state_token": state_token,
            "payload": payload,
        }
        if arguments is None
        else arguments
    )
    return {
        "id": f"lg_interrupt_{interrupt_id}",
        "type": "function",
        "function": {
            "name": "langgraph_interrupt",
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


def _completion(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, object]] | None = None,
) -> ChatCompletion:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "interruptible",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


def _ledger_step(
    tool_calls: list[dict[str, object]],
    *,
    status: str = "pending",
    model_id: str = "lgos-a/hitl",
    raw_ledger: object | None = None,
) -> dict[str, Any]:
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    if raw_ledger is None:
        ledger_dict: dict[str, object] = {
            "schema_version": 1,
            "status": status,
        }
        if status == "pending":
            ledger_dict.update(
                {
                    "model_id": model_id,
                    "assistant_message": assistant_message,
                }
            )
        ledger: object = ledger_dict
    else:
        ledger = raw_ledger
    return {
        "id": "persisted-ledger-message",
        "type": "assistant_message",
        "name": "Assistant",
        "output": "",
        "createdAt": "2026-08-10T12:00:00Z",
        "metadata": {
            "lgos_chainlit.exclude_from_model_context": True,
            "lgos_chainlit.hitl_interrupt_ledger": ledger,
        },
    }


def test_model_support_is_read_from_openai_extension() -> None:
    model = Model(
        id="interruptible",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "description": "DUMMY",
            "features": ["interrupts"],
        },
    )

    assert model_supports(model, GraphFeature.INTERRUPTS)


def test_model_support_rejects_unknown_extension_version() -> None:
    model = Model(
        id="interruptible",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 2,
            "features": ["interrupts"],
        },
    )

    assert not model_supports(model, GraphFeature.INTERRUPTS)


def test_model_metadata_rejects_a_blank_description() -> None:
    model = Model(
        id="interruptible",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "description": "   ",
            "features": ["interrupts"],
        },
    )

    assert model_description(model) is None
    assert not model_supports(model, GraphFeature.INTERRUPTS)


async def test_chat_profile_warns_when_description_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    monkeypatch.setattr(
        hitl,
        "retrieve_model",
        AsyncMock(
            return_value=Model(
                id="interruptible",
                object="model",
                created=1,
                owned_by="test",
                langgraph_openai_serve={
                    "schema_version": 1,
                    "features": ["interrupts"],
                },
            )
        ),
    )

    profiles = await hitl.set_chat_profiles(None)

    assert len(profiles) == 1
    assert profiles[0].markdown_description == hitl.LIMITED_FUNCTIONALITY_MESSAGE


async def test_chat_profile_keeps_the_provider_qualified_model(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    monkeypatch.setattr(hitl.settings, "HITL_MODEL", "lgos-b/interruptible")
    monkeypatch.setattr(
        hitl,
        "retrieve_model",
        AsyncMock(
            return_value=Model(
                id="interruptible",
                object="model",
                created=1,
                owned_by="test",
                langgraph_openai_serve={
                    "schema_version": 1,
                    "description": "DUMMY",
                    "features": ["interrupts"],
                },
            )
        ),
    )

    profiles = await hitl.set_chat_profiles(None)

    assert [profile.name for profile in profiles] == ["lgos-b/interruptible"]
    assert profiles[0].markdown_description == "DUMMY"


async def test_chat_profile_rejects_a_valid_non_interrupt_model(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    monkeypatch.setattr(
        hitl,
        "retrieve_model",
        AsyncMock(
            return_value=Model(
                id="simple",
                object="model",
                created=1,
                owned_by="test",
                langgraph_openai_serve={
                    "schema_version": 1,
                    "description": "DUMMY",
                    "features": [],
                },
            )
        ),
    )

    with pytest.raises(RuntimeError, match="does not advertise interrupt support"):
        await hitl.set_chat_profiles(None)


@pytest.mark.parametrize("mixed", [False, True], ids=["ordinary", "mixed"])
async def test_resolve_interrupts_reports_unsupported_tool_call_batches(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
    mixed: bool,
) -> None:
    ordinary_call = {
        "id": "call_other",
        "type": "function",
        "function": {"name": "other_tool", "arguments": "{}"},
    }
    tool_calls = [ordinary_call]
    if mixed:
        tool_calls.insert(
            0,
            _interrupt_call("interrupt-1", {"question": "Approve?"}),
        )
    send_ui_message = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", send_ui_message)

    await hitl.resolve_interrupts(
        assistant_message=_completion(tool_calls=tool_calls).choices[0].message,
        model_id="lgos-a/hitl",
    )

    send_ui_message.assert_awaited_once_with("Received an unsupported tool-call batch.")


@pytest.mark.parametrize(
    ("payload", "prompt"),
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
        pytest.param(
            {
                "question": "Approve?",
                "request": "Refund ORDER-123.",
                "choices": ["approve", "reject"],
            },
            "Approve?\n\nRequest: Refund ORDER-123.",
            id="object",
        ),
        pytest.param(
            {"amount": 42},
            'Approve this action?\n\n{\n  "amount": 42\n}',
            id="object-without-prompt-fields",
        ),
    ],
)
def test_interrupt_payload_accepts_and_renders_every_json_value(
    hitl: Any,
    payload: object,
    prompt: str,
) -> None:
    message = (
        _completion(tool_calls=[_interrupt_call("interrupt-1", payload)])
        .choices[0]
        .message
    )

    tool_calls = hitl.interrupt_tool_calls(message)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert hitl.interrupt_payload(tool_calls[0]) == payload
    assert hitl.interrupt_prompt(payload) == prompt


def test_interrupt_payload_rejects_a_missing_payload(
    hitl: Any,
) -> None:
    arguments = {
        "run_id": RUN_ID,
        "state_token": STATE_TOKEN,
    }
    message = (
        _completion(
            tool_calls=[
                _interrupt_call("interrupt-1", {}, arguments=arguments),
            ]
        )
        .choices[0]
        .message
    )

    with pytest.raises(ValueError, match="contain a payload"):
        hitl.interrupt_payload(message.tool_calls[0])


async def test_handle_message_resumes_all_interrupts_once(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    payloads = [
        {"question": "Approve refund?", "request": "ORDER-123"},
        {"question": "Approve notification?", "request": "Email customer"},
    ]
    interrupt_calls = [
        _interrupt_call("interrupt-1", payloads[0]),
        _interrupt_call("interrupt-2", payloads[1]),
    ]
    create_completion = AsyncMock(
        side_effect=[
            _completion(tool_calls=interrupt_calls),
            _completion("Both decisions applied."),
        ]
    )
    _, created_messages, writes = _install_message_factory(monkeypatch, hitl)
    decisions = iter(["approve", "reject"])

    async def decide(_: object) -> str:
        assert writes[0][0] == "send"
        assert writes[0][2][hitl.INTERRUPT_LEDGER_METADATA_KEY]["status"] == (
            hitl.PENDING_LEDGER_STATUS
        )
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
    first_call = _interrupt_call("interrupt-1", {"question": "First?"})
    second_call = _interrupt_call(
        "interrupt-2",
        {"question": "Second?"},
        state_token="state-token-2",
    )
    create_completion = AsyncMock(
        side_effect=[
            _completion(tool_calls=[first_call]),
            _completion(tool_calls=[second_call]),
            _completion("Done."),
        ]
    )
    _, created_messages, writes = _install_message_factory(monkeypatch, hitl)
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
        return_value=_completion(
            tool_calls=[
                _interrupt_call("interrupt-1", {"question": "First?"}),
                _interrupt_call("interrupt-2", {"question": "Second?"}),
            ]
        )
    )
    ask_for_resume = AsyncMock(return_value=None)
    _, created_messages, writes = _install_message_factory(monkeypatch, hitl)
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
    interrupt_call = _interrupt_call(
        "interrupt-1",
        {"question": "Approve?"},
    )
    create_completion = AsyncMock(
        side_effect=[
            _completion(tool_calls=[interrupt_call]),
            _completion("Approved."),
        ]
    )
    _, _, writes = _install_message_factory(monkeypatch, hitl)
    session: dict[str, object] = {}
    user_session = SimpleNamespace(
        get=lambda key, default=None: session.get(key, default),
        set=lambda key, value: session.__setitem__(key, value),
    )
    text_only_chat_messages = Mock(return_value=[{"role": "user", "content": "Start."}])
    monkeypatch.setattr(hitl.cl, "user_session", user_session)
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
    trigger_message = _recording_message(
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
        "Resolve the pending approval before starting another request."
    )
    assert session[hitl.PENDING_LEDGER_SESSION_KEY] is None
    assert [write[0] for write in writes] == ["send", "update", "update", "send"]
    assert [write[0] for write in trigger_writes] == ["update"]
    assert trigger_writes[0][2] == {"lgos_chainlit.exclude_from_model_context": True}


async def test_on_chat_resume_restores_ledger_for_the_next_user_interaction(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    interrupt_calls = [
        _interrupt_call("interrupt-1", {"question": "First?"}),
        _interrupt_call("interrupt-2", {"question": "Second?"}),
    ]
    step = _ledger_step(interrupt_calls)
    restored_writes: list[tuple[str, Mock, dict[str, object] | None]] = []
    restored_message = _recording_message(
        metadata=step["metadata"],
        writes=restored_writes,
    )
    factory, created_messages, writes = _install_message_factory(
        monkeypatch,
        hitl,
        restored=restored_message,
    )
    create_completion = AsyncMock(return_value=_completion("Resumed."))
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
    step = _ledger_step([], raw_ledger={"schema_version": 999})
    factory, _, _ = _install_message_factory(monkeypatch, hitl)
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
    step = _ledger_step(
        [_interrupt_call("interrupt-1", {"question": "Approve?"})],
        status="completed",
    )
    factory, _, _ = _install_message_factory(monkeypatch, hitl)
    create_completion = AsyncMock()
    ask_for_resume = AsyncMock()
    monkeypatch.setattr(hitl, "create_completion", create_completion)
    monkeypatch.setattr(hitl, "ask_for_resume", ask_for_resume)
    monkeypatch.setattr(
        hitl,
        "_warn_if_model_metadata_is_missing",
        AsyncMock(),
    )

    await hitl.on_chat_resume({"steps": [step]})

    factory.from_dict.assert_not_called()
    create_completion.assert_not_awaited()
    ask_for_resume.assert_not_awaited()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        pytest.param(None, "Approval timed out.", id="timeout"),
        pytest.param(
            {"payload": {}},
            "No approval decision was received.",
            id="malformed-action-result",
        ),
    ],
)
async def test_ask_for_resume_never_defaults_missing_decision_to_reject(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
    response: dict[str, object] | None,
    message: str,
) -> None:
    action_message = Mock(metadata=None, send=AsyncMock(return_value=response))
    send_ui_message = AsyncMock()
    tool_call = (
        _completion(
            tool_calls=[_interrupt_call("interrupt-1", {"question": "Approve?"})]
        )
        .choices[0]
        .message.tool_calls[0]
    )
    monkeypatch.setattr(hitl.cl, "AskActionMessage", Mock(return_value=action_message))
    monkeypatch.setattr(hitl, "mark_model_context_excluded", Mock())
    monkeypatch.setattr(hitl, "send_ui_message", send_ui_message)

    decision = await hitl.ask_for_resume(tool_call)

    assert decision is None
    send_ui_message.assert_awaited_once_with(message)


async def test_create_completion_sends_no_long_lived_run_metadata(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    create = AsyncMock(return_value=_completion("Done."))
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(get=lambda _: "interruptible"),
    )
    monkeypatch.setattr(hitl, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(hitl.openai_client.chat.completions, "create", create)

    await hitl.create_completion([{"role": "user", "content": "Hello"}])

    create.assert_awaited_once_with(
        model="interruptible",
        messages=[{"role": "user", "content": "Hello"}],
        user="demo-user",
    )
    assert hitl.openai_client.max_retries == 0


async def test_completion_errors_are_excluded_from_model_context(
    monkeypatch: pytest.MonkeyPatch,
    hitl: Any,
) -> None:
    error_message = Mock(metadata=None, send=AsyncMock())
    monkeypatch.setattr(hitl, "text_only_chat_messages", Mock(return_value=[]))
    monkeypatch.setattr(
        hitl,
        "create_completion",
        AsyncMock(side_effect=RuntimeError("backend unavailable")),
    )
    monkeypatch.setattr(hitl.cl, "Message", Mock(return_value=error_message))

    await hitl.on_message(Mock())

    assert error_message.metadata == {"lgos_chainlit.exclude_from_model_context": True}
    error_message.send.assert_awaited_once_with()
