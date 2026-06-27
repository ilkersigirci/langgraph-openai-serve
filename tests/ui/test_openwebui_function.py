import json
from typing import Any

import pytest
from demo.ui.openwebui.hitl_function import Pipe
from openai.types.chat import ChatCompletion

from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME


def _completion(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="interruptible-approval",
        choices=[{"index": 0, "finish_reason": "stop", "message": message}],
    )


def _interrupt_response() -> ChatCompletion:
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": INTERRUPT_TOOL_NAME,
            "arguments": json.dumps(
                {
                    "payload": {
                        "question": "Approve?",
                        "request": "Refund order ORDER-123",
                    }
                }
            ),
        },
    }
    return _completion(tool_calls=[tool_call])


def _malformed_interrupt_response() -> ChatCompletion:
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": INTERRUPT_TOOL_NAME,
            "arguments": json.dumps([]),
        },
    }
    return _completion(tool_calls=[tool_call])


@pytest.mark.anyio
async def test_openwebui_function_asks_for_confirmation_and_approves() -> None:
    pipe = Pipe()
    chat_calls = []
    events = []

    async def chat(messages, thread_id):
        chat_calls.append((messages, thread_id))
        if len(chat_calls) == 1:
            return _interrupt_response()
        return _completion("Approved agent action: Refund order ORDER-123")

    async def event_call(event):
        events.append(event)
        return True

    pipe._chat = chat  # type: ignore[method-assign]

    response = await pipe.pipe(
        body={"messages": [{"role": "user", "content": "Refund order ORDER-123"}]},
        __event_call__=event_call,
        __metadata__={"chat_id": "chat-1", "session_id": "session-1"},
    )

    assert response == "Approved agent action: Refund order ORDER-123"
    assert events == [
        {
            "type": "confirmation",
            "data": {
                "title": "Approve?",
                "message": "Refund order ORDER-123",
            },
        }
    ]
    resume_messages, thread_id = chat_calls[1]
    assert resume_messages[0] == {
        "role": "user",
        "content": "Refund order ORDER-123",
    }
    assert resume_messages[1]["tool_calls"][0]["id"] == "call-1"
    assert resume_messages[2]["tool_call_id"] == "call-1"
    assert json.loads(resume_messages[2]["content"]) == {
        "resume": "approve"
    }
    assert thread_id == "openwebui:function:chat-1"


@pytest.mark.anyio
async def test_openwebui_function_rejects_when_confirmation_declines() -> None:
    pipe = Pipe()
    chat_calls = []

    async def chat(messages, thread_id):
        del thread_id
        chat_calls.append(messages)
        if len(chat_calls) == 1:
            return _interrupt_response()
        return _completion("Rejected agent action: Refund order ORDER-123")

    async def event_call(event):
        return False

    pipe._chat = chat  # type: ignore[method-assign]

    response = await pipe.pipe(
        body={"messages": [{"role": "user", "content": "Refund order ORDER-123"}]},
        __event_call__=event_call,
        __metadata__={"chat_id": "chat-1"},
    )

    assert response == "Rejected agent action: Refund order ORDER-123"
    assert json.loads(chat_calls[1][-1]["content"]) == {
        "resume": "reject"
    }


@pytest.mark.anyio
async def test_openwebui_function_handles_malformed_interrupt_arguments() -> None:
    pipe = Pipe()
    events = []

    async def chat(messages, thread_id):
        del messages, thread_id
        if not events:
            return _malformed_interrupt_response()
        return _completion("Approved agent action.")

    async def event_call(event):
        events.append(event)
        return True

    pipe._chat = chat  # type: ignore[method-assign]

    response = await pipe.pipe(
        body={"messages": [{"role": "user", "content": "Refund order ORDER-123"}]},
        __event_call__=event_call,
        __metadata__={"chat_id": "chat-1"},
    )

    assert response == "Approved agent action."
    assert events == [
        {
            "type": "confirmation",
            "data": {
                "title": "Approve this agent action?",
                "message": "{}",
            },
        }
    ]


@pytest.mark.anyio
async def test_openwebui_function_sends_thread_id_as_openai_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    completion = _completion("ok")

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return completion

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("demo.ui.openwebui.hitl_function.AsyncOpenAI", FakeOpenAI)

    pipe = Pipe()
    response = await pipe._chat(
        [{"role": "user", "content": "hello"}],
        "openwebui:function:chat-1",
    )

    assert response == completion
    assert captured["metadata"] == {
        "langgraph_thread_id": "openwebui:function:chat-1"
    }
    assert "extra_body" not in captured
