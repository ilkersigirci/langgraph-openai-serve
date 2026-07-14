import importlib
import json
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionMessage

from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME


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
