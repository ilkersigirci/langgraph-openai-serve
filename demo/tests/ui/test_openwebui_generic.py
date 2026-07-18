import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from demo.ui.openwebui.functions.generic import Pipe
from openai.lib.streaming.chat import ContentDeltaEvent
from openai.types.chat import ChatCompletion

from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME

pytestmark = pytest.mark.anyio

USER_REQUEST = "Refund order ORDER-123"
THREAD_ID = "openwebui:function:chat-1"
MODEL_ID = "interruptible-approval"
MARKDOWN_DELTAS = (
    "Read the [source](https://example.com/source), ",
    "view ![diagram](https://example.com/diagram.png), ",
    "and follow the [audio link](https://example.com/overview.mp3).",
)
MARKDOWN_RESPONSE = "".join(MARKDOWN_DELTAS)
INTERRUPT_PAYLOAD = {
    "question": "Approve?",
    "request": USER_REQUEST,
}


class ScriptedStream:
    def __init__(
        self,
        deltas: Sequence[str],
        completion: ChatCompletion,
    ) -> None:
        self._deltas = deltas
        self._completion = completion

    async def __aiter__(self) -> AsyncIterator[ContentDeltaEvent]:
        snapshot = ""
        for delta in self._deltas:
            snapshot += delta
            yield ContentDeltaEvent(
                type="content.delta",
                delta=delta,
                snapshot=snapshot,
                parsed=None,
            )

    async def get_final_completion(self) -> ChatCompletion:
        return self._completion


class ScriptedChat:
    def __init__(
        self,
        *steps: tuple[Sequence[str], ChatCompletion],
    ) -> None:
        self._steps = steps
        self.calls: list[tuple[list[dict[str, Any]], str, str]] = []

    @asynccontextmanager
    async def __call__(
        self,
        messages: list[dict[str, Any]],
        thread_id: str,
        model_id: str,
    ) -> AsyncIterator[ScriptedStream]:
        step_index = len(self.calls)
        self.calls.append((messages, thread_id, model_id))
        if step_index >= len(self._steps):
            raise AssertionError(f"Unexpected chat call {step_index + 1}")

        deltas, completion = self._steps[step_index]
        yield ScriptedStream(deltas, completion)


async def _collect_response(
    pipe_response: AsyncIterator[str | dict[str, Any]],
) -> list[str | dict[str, Any]]:
    return [chunk async for chunk in pipe_response]


def _body(
    content: str,
    model: str = f"generic.{MODEL_ID}",
    *,
    stream: bool = True,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }


def _completion(
    content: str = "",
    *,
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
            "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


def _interrupt_response(arguments: object | None = None) -> ChatCompletion:
    arguments = {"payload": INTERRUPT_PAYLOAD} if arguments is None else arguments
    return _completion(
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": INTERRUPT_TOOL_NAME,
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    )


def _citation_response() -> ChatCompletion:
    citation_text = "source"
    start = MARKDOWN_RESPONSE.index(citation_text)
    return _completion(
        MARKDOWN_RESPONSE,
        annotations=[
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": start,
                    "end_index": start + len(citation_text) - 1,
                    "title": "Example source",
                    "url": "https://example.com/source",
                },
            }
        ],
    )


async def test_pipe_lists_registered_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="interruptible-approval"),
            SimpleNamespace(id="lgos-rag"),
        ]
    )
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(
        "demo.ui.openwebui.functions.generic.AsyncOpenAI",
        client_factory,
    )

    models = await Pipe().pipes()

    assert models == [
        {
            "id": "interruptible-approval",
            "name": "Generic / interruptible-approval",
        },
        {"id": "lgos-rag", "name": "Generic / lgos-rag"},
    ]
    client_factory.assert_called_once_with(
        base_url="http://lgos-demo-api:8000/v1",
        api_key="DUMMY",
        timeout=30,
    )
    client.models.list.assert_awaited_once_with()
    client.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_preserves_dots_in_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat((("ok",), _completion("ok")))
    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("hello", model="generic.graph.v2"),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == ["ok"]
    assert chat.calls == [
        ([{"role": "user", "content": "hello"}], THREAD_ID, "graph.v2")
    ]


async def test_pipe_rejects_unqualified_model_id() -> None:
    chunks = await _collect_response(Pipe().pipe(body=_body("hello", model="graph")))

    assert chunks == ["Open WebUI did not provide a valid model ID."]


@pytest.mark.parametrize(
    ("approved", "decision", "answer_deltas"),
    [
        pytest.param(
            True,
            "approve",
            ("Approved agent action: ", USER_REQUEST),
            id="approve",
        ),
        pytest.param(
            False,
            "reject",
            (f"Rejected agent action: {USER_REQUEST}",),
            id="reject",
        ),
    ],
)
async def test_pipe_resumes_confirmed_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    approved: bool,
    decision: str,
    answer_deltas: tuple[str, ...],
) -> None:
    pipe = Pipe()
    chat = ScriptedChat(
        ((), _interrupt_response()),
        (answer_deltas, _completion("".join(answer_deltas))),
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return approved

    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body(USER_REQUEST),
            __event_call__=confirm,
            __metadata__={"chat_id": "chat-1", "session_id": "session-1"},
        )
    )

    assert chunks == list(answer_deltas)
    assert events == [
        {
            "type": "confirmation",
            "data": {"title": "Approve?", "message": USER_REQUEST},
        }
    ]
    (
        (initial_messages, initial_thread_id, initial_model_id),
        (resume_messages, resume_thread_id, resume_model_id),
    ) = chat.calls
    assert initial_messages == [{"role": "user", "content": USER_REQUEST}]
    assert resume_messages[0] == initial_messages[0]
    assert resume_messages[1]["tool_calls"][0]["id"] == "call-1"
    assert json.loads(resume_messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "payload": INTERRUPT_PAYLOAD
    }
    assert resume_messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": json.dumps({"resume": decision}),
    }
    assert (initial_thread_id, resume_thread_id) == (THREAD_ID, THREAD_ID)
    assert (initial_model_id, resume_model_id) == (MODEL_ID, MODEL_ID)


async def test_pipe_uses_fallback_confirmation_for_malformed_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat(
        ((), _interrupt_response([])),
        (("Approved ", "agent action."), _completion("Approved agent action.")),
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return True

    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body(USER_REQUEST),
            __event_call__=confirm,
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
    assert json.loads(chat.calls[1][0][-1]["content"]) == {"resume": "approve"}


async def test_chat_sends_model_and_thread_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    messages = [{"role": "user", "content": "hello"}]
    completion = _completion("ok")
    stream = ScriptedStream(("ok",), completion)
    stream_context = AsyncMock()
    stream_context.__aenter__.return_value = stream
    client = AsyncMock()
    client.__aenter__.return_value = client
    stream_factory = Mock(return_value=stream_context)
    client.chat.completions.stream = stream_factory
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(pipe, "_client", client_factory)

    async with pipe._chat(messages, THREAD_ID, "graph.with.dots") as response_stream:
        deltas = [
            event.delta
            async for event in response_stream
            if isinstance(event, ContentDeltaEvent)
        ]
        response = await response_stream.get_final_completion()

    assert response == completion
    assert deltas == ["ok"]
    client_factory.assert_called_once_with()
    stream_factory.assert_called_once_with(
        model="graph.with.dots",
        messages=messages,
        metadata={"langgraph_thread_id": THREAD_ID},
    )
    stream_context.__aexit__.assert_awaited_once_with(None, None, None)
    client.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_streams_markdown_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat((MARKDOWN_DELTAS, _completion(MARKDOWN_RESPONSE)))
    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("Cite this", model="generic.lgos-rag"),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == list(MARKDOWN_DELTAS)
    assert chat.calls[0][1:] == (THREAD_ID, "lgos-rag")


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(True, id="streaming"),
        pytest.param(False, id="non-streaming"),
    ],
)
async def test_pipe_forwards_annotations_only_when_streaming(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    pipe = Pipe()
    completion = _citation_response()
    chat = ScriptedChat(((MARKDOWN_RESPONSE,), completion))
    monkeypatch.setattr(pipe, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body(
                "Cite this",
                model="generic.citation-events",
                stream=stream,
            ),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    expected: list[str | dict[str, Any]] = [MARKDOWN_RESPONSE]
    if stream:
        annotation = completion.choices[0].message.annotations[0]
        expected.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "annotations": [annotation.model_dump(mode="json")],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
    assert chunks == expected
    assert chat.calls[0][1:] == (THREAD_ID, "citation-events")
