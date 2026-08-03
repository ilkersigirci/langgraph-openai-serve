import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types import Model
from openai.types.chat import ChatCompletionMessage

from lgos_chainlit.lgos_protocol import (
    GraphFeature,
    model_description,
    model_supports,
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


@pytest.mark.anyio
async def test_chat_profile_warns_when_description_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    hitl = importlib.import_module("lgos_chainlit.hitl")
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


@pytest.mark.anyio
async def test_chat_profile_keeps_the_provider_qualified_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    hitl = importlib.import_module("lgos_chainlit.hitl")
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


@pytest.mark.anyio
async def test_chat_profile_rejects_a_valid_non_interrupt_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    hitl = importlib.import_module("lgos_chainlit.hitl")
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


def test_helpers_extract_and_preserve_the_interrupt_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    hitl = importlib.import_module("lgos_chainlit.hitl")
    payload = {
        "question": "Approve?",
        "request": "Refund ORDER-123.",
        "choices": ["approve", "reject"],
    }
    interrupt_call = {
        "id": "call_interrupt",
        "type": "function",
        "function": {
            "name": "langgraph_interrupt",
            "arguments": json.dumps({"payload": payload}),
        },
    }
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_other",
                "type": "function",
                "function": {"name": "other_tool", "arguments": "{}"},
            },
            interrupt_call,
        ],
    )

    tool_call = hitl.interrupt_tool_call(message)

    assert tool_call is not None
    assert tool_call.id == "call_interrupt"
    assert hitl.interrupt_payload(tool_call) == payload
    assert hitl.assistant_tool_call_message(message, tool_call) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [interrupt_call],
    }


@pytest.mark.anyio
async def test_completion_errors_are_excluded_from_model_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    hitl = importlib.import_module("lgos_chainlit.hitl")
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
