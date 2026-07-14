import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types import Model
from openai.types.chat import ChatCompletionMessage

from langgraph_openai_serve import GraphFeature
from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME


def test_model_support_is_read_from_openai_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("CHAINLIT_ENV_FILE", str(tmp_path / "missing.env"))
    hitl = importlib.import_module("demo.ui.chainlit_ui.hitl")
    model = Model(
        id="interruptible",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": ["interrupts"],
        },
    )

    assert hitl.model_supports(model, GraphFeature.INTERRUPTS)


def test_model_support_rejects_unknown_extension_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("CHAINLIT_ENV_FILE", str(tmp_path / "missing.env"))
    hitl = importlib.import_module("demo.ui.chainlit_ui.hitl")
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

    assert not hitl.model_supports(model, GraphFeature.INTERRUPTS)


@pytest.mark.anyio
async def test_chat_profiles_fail_when_feature_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("CHAINLIT_ENV_FILE", str(tmp_path / "missing.env"))
    hitl = importlib.import_module("demo.ui.chainlit_ui.hitl")
    monkeypatch.setattr(
        hitl.client.models,
        "list",
        AsyncMock(return_value=SimpleNamespace(data=[])),
    )

    with pytest.raises(RuntimeError, match="pass-through configured to target LGOS"):
        await hitl.set_chat_profiles()


def test_helpers_extract_and_preserve_the_interrupt_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("CHAINLIT_ENV_FILE", str(tmp_path / "missing.env"))
    hitl = importlib.import_module("demo.ui.chainlit_ui.hitl")
    payload = {
        "question": "Approve?",
        "request": "Refund ORDER-123.",
        "choices": ["approve", "reject"],
    }
    interrupt_call = {
        "id": "call_interrupt",
        "type": "function",
        "function": {
            "name": INTERRUPT_TOOL_NAME,
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
