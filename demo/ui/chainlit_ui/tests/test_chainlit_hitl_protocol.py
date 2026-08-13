from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types import Model

from lgos_chainlit.lgos_protocol import (
    GraphFeature,
    model_description,
    model_supports,
)

from .chainlit_hitl_support import (
    RUN_ID,
    STATE_TOKEN,
    completion,
    interrupt_call,
)


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
            interrupt_call("interrupt-1", {"question": "Approve?"}),
        )
    send_ui_message = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", send_ui_message)

    await hitl.resolve_interrupts(
        assistant_message=completion(tool_calls=tool_calls).choices[0].message,
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
        completion(tool_calls=[interrupt_call("interrupt-1", payload)])
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
        completion(
            tool_calls=[
                interrupt_call("interrupt-1", {}, arguments=arguments),
            ]
        )
        .choices[0]
        .message
    )

    with pytest.raises(ValueError, match="contain a payload"):
        hitl.interrupt_payload(message.tool_calls[0])


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
        completion(tool_calls=[interrupt_call("interrupt-1", {"question": "Approve?"})])
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
    create = AsyncMock(return_value=completion("Done."))
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
