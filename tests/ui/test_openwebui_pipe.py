import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from demo.ui.openwebui.openwebui_pipe import Pipe
from openai.lib.streaming.chat import ContentDeltaEvent
from openai.types.chat import ChatCompletion

from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME


class FakeStream:
    def __init__(
        self,
        deltas: list[str],
        completion: ChatCompletion,
    ) -> None:
        self.deltas = deltas
        self.completion = completion

    async def __aiter__(self) -> AsyncIterator[ContentDeltaEvent]:
        snapshot = ""
        for delta in self.deltas:
            snapshot += delta
            yield ContentDeltaEvent(
                type="content.delta",
                delta=delta,
                snapshot=snapshot,
                parsed=None,
            )

    async def get_final_completion(self) -> ChatCompletion:
        return self.completion


async def _collect_response(
    pipe_response: AsyncIterator[str | dict[str, Any]],
) -> list[str | dict[str, Any]]:
    return [chunk async for chunk in pipe_response]


def _body(
    content: str,
    model: str = "openwebui_pipe.interruptible-approval",
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }


def _completion(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if annotations is not None:
        message["annotations"] = annotations

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "interruptible-approval",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
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


def _citation_response() -> ChatCompletion:
    return _completion(
        "Cited answer [1]",
        annotations=[
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 13,
                    "end_index": 16,
                    "title": "Example source",
                    "url": "https://example.com/source",
                },
            }
        ],
    )


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
async def test_openwebui_pipe_discovers_registered_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeModels:
        async def list(self):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(id="interruptible-approval"),
                    SimpleNamespace(id="lgos-rag"),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr(
        "demo.ui.openwebui.openwebui_pipe.AsyncOpenAI",
        FakeOpenAI,
    )

    models = await Pipe().pipes()

    assert models == [
        {"id": "interruptible-approval", "name": "interruptible-approval"},
        {"id": "lgos-rag", "name": "lgos-rag"},
    ]
    assert captured["base_url"] == "http://lgos-demo-api:8000/v1"
    assert captured["api_key"] == "DUMMY"


def test_openwebui_pipe_preserves_dots_in_selected_model_id() -> None:
    assert Pipe()._model_id({"model": "openwebui_pipe.graph.v2"}) == "graph.v2"


@pytest.mark.anyio
async def test_openwebui_pipe_rejects_an_unqualified_model_id() -> None:
    chunks = await _collect_response(Pipe().pipe(body=_body("hello", model="graph")))

    assert chunks == ["Open WebUI did not provide a valid model ID."]


@pytest.mark.anyio
async def test_openwebui_pipe_asks_for_confirmation_and_approves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat_calls = []
    events = []

    @asynccontextmanager
    async def chat(messages, thread_id, model_id):
        chat_calls.append((messages, thread_id, model_id))
        if len(chat_calls) == 1:
            yield FakeStream([], _interrupt_response())
            return
        answer = "Approved agent action: Refund order ORDER-123"
        yield FakeStream(
            ["Approved agent action: ", "Refund order ORDER-123"],
            _completion(answer),
        )

    async def event_call(event):
        events.append(event)
        return True

    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("Refund order ORDER-123"),
            __event_call__=event_call,
            __metadata__={"chat_id": "chat-1", "session_id": "session-1"},
        )
    )

    assert chunks == ["Approved agent action: ", "Refund order ORDER-123"]
    assert events == [
        {
            "type": "confirmation",
            "data": {
                "title": "Approve?",
                "message": "Refund order ORDER-123",
            },
        }
    ]
    resume_messages, thread_id, model_id = chat_calls[1]
    assert resume_messages[0] == {
        "role": "user",
        "content": "Refund order ORDER-123",
    }
    assert resume_messages[1]["tool_calls"][0]["id"] == "call-1"
    assert resume_messages[2]["tool_call_id"] == "call-1"
    assert json.loads(resume_messages[2]["content"]) == {"resume": "approve"}
    assert thread_id == "openwebui:function:chat-1"
    assert model_id == "interruptible-approval"
    assert chat_calls[0][2] == model_id


@pytest.mark.anyio
async def test_openwebui_pipe_rejects_when_confirmation_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat_calls = []

    @asynccontextmanager
    async def chat(messages, thread_id, model_id):
        del thread_id
        chat_calls.append((messages, model_id))
        if len(chat_calls) == 1:
            yield FakeStream([], _interrupt_response())
            return
        answer = "Rejected agent action: Refund order ORDER-123"
        yield FakeStream([answer], _completion(answer))

    async def event_call(event):
        return False

    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("Refund order ORDER-123"),
            __event_call__=event_call,
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == ["Rejected agent action: Refund order ORDER-123"]
    assert json.loads(chat_calls[1][0][-1]["content"]) == {"resume": "reject"}
    assert {model_id for _, model_id in chat_calls} == {"interruptible-approval"}


@pytest.mark.anyio
async def test_openwebui_pipe_handles_malformed_interrupt_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    events = []

    @asynccontextmanager
    async def chat(messages, thread_id, model_id):
        del messages, thread_id, model_id
        if not events:
            yield FakeStream([], _malformed_interrupt_response())
            return
        yield FakeStream(
            ["Approved ", "agent action."],
            _completion("Approved agent action."),
        )

    async def event_call(event):
        events.append(event)
        return True

    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("Refund order ORDER-123"),
            __event_call__=event_call,
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == ["Approved ", "agent action."]
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
async def test_openwebui_pipe_sends_selected_model_and_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    completion = _completion("ok")

    class FakeStreamManager:
        async def __aenter__(self):
            return FakeStream(["ok"], completion)

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeCompletions:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStreamManager()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs
            self.chat = FakeChat()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr(
        "demo.ui.openwebui.openwebui_pipe.AsyncOpenAI",
        FakeOpenAI,
    )

    pipe = Pipe()
    async with pipe._chat(
        [{"role": "user", "content": "hello"}],
        "openwebui:function:chat-1",
        "graph.with.dots",
    ) as stream:
        deltas = [
            event.delta
            async for event in stream
            if isinstance(event, ContentDeltaEvent)
        ]
        response = await stream.get_final_completion()

    assert response == completion
    assert deltas == ["ok"]
    assert captured["model"] == "graph.with.dots"
    assert captured["metadata"] == {"langgraph_thread_id": "openwebui:function:chat-1"}
    assert "stream" not in captured
    assert "extra_body" not in captured


@pytest.mark.anyio
async def test_openwebui_pipe_forwards_openai_citation_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()

    @asynccontextmanager
    async def chat(messages, thread_id, model_id):
        del messages, thread_id, model_id
        yield FakeStream(
            ["Cited ", "answer ", "[1]"],
            _citation_response(),
        )

    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("Cite this", model="openwebui_pipe.lgos-rag"),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == [
        "Cited ",
        "answer ",
        "[1]",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "start_index": 13,
                                    "end_index": 16,
                                    "title": "Example source",
                                    "url": "https://example.com/source",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                },
            ]
        },
    ]
