import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.lib.streaming.chat import ChunkEvent, ContentDeltaEvent
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from lgos_openwebui.functions import generic
from lgos_openwebui.functions.generic import Pipe

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
        self.runtime_metadata_calls: list[dict[str, str] | None] = []
        self.timeouts: list[float] = []

    @asynccontextmanager
    async def __call__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        messages: list[dict[str, Any]],
        thread_id: str,
        model_id: str,
        runtime_metadata: dict[str, str] | None = None,
    ) -> AsyncIterator[ScriptedStream]:
        step_index = len(self.calls)
        self.calls.append((messages, thread_id, model_id))
        self.runtime_metadata_calls.append(runtime_metadata)
        self.timeouts.append(timeout)
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
                    "name": "langgraph_interrupt",
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


def test_api_key_valve_uses_password_input() -> None:
    api_key_schema = Pipe.Valves.model_json_schema()["properties"]["OPENAI_API_KEY"]

    assert api_key_schema["input"] == {"type": "password"}


async def test_pipe_lists_registered_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    pipe.valves.OPENAI_API_TIMEOUT = 45
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="lgos-a/interruptible-approval"),
            SimpleNamespace(id="lgos-b/lgos-rag"),
        ]
    )
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(
        "lgos_openwebui.functions.generic.AsyncOpenAI",
        client_factory,
    )

    models = await pipe.pipes()

    assert models == [
        {
            "id": "lgos-a/interruptible-approval",
            "name": "Generic / lgos-a/interruptible-approval",
        },
        {"id": "lgos-b/lgos-rag", "name": "Generic / lgos-b/lgos-rag"},
    ]
    client_factory.assert_called_once_with(
        base_url="http://bifrost:8080/v1",
        api_key="DUMMY",
        timeout=45,
    )
    client.models.list.assert_awaited_once_with()
    client.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_preserves_dots_in_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat((("ok",), _completion("ok")))
    monkeypatch.setattr(generic, "_chat", chat)

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

    monkeypatch.setattr(generic, "_chat", chat)

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

    monkeypatch.setattr(generic, "_chat", chat)

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
    monkeypatch.setattr(generic, "_client", client_factory)

    async with generic._chat(
        base_url=pipe.valves.OPENAI_API_BASE_URL,
        api_key=pipe.valves.OPENAI_API_KEY,
        timeout=pipe.valves.OPENAI_API_TIMEOUT,
        messages=messages,
        thread_id=THREAD_ID,
        model_id="lgos-a/graph.with.dots",
        runtime_metadata={"langgraph_runtime_settings": '{"mode":"detailed"}'},
    ) as response_stream:
        deltas = [
            event.delta
            async for event in response_stream
            if isinstance(event, ContentDeltaEvent)
        ]
        response = await response_stream.get_final_completion()

    assert response == completion
    assert deltas == ["ok"]
    client_factory.assert_called_once_with(
        base_url=pipe.valves.OPENAI_API_BASE_URL,
        api_key=pipe.valves.OPENAI_API_KEY,
        timeout=pipe.valves.OPENAI_API_TIMEOUT,
    )
    stream_factory.assert_called_once_with(
        model="graph.with.dots",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=messages,
        metadata={
            "langgraph_thread_id": THREAD_ID,
            "langgraph_stream_events": "v1",
            "langgraph_runtime_settings": '{"mode":"detailed"}',
        },
    )
    stream_context.__aexit__.assert_awaited_once_with(None, None, None)
    client.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_forwards_changed_chat_variables_as_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    model = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "client_settings": {
                    "schema_version": 1,
                    "defaults": {
                        "use_history": False,
                        "audience": "general",
                    },
                },
            }
        }
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = model
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(generic, "_client", client_factory)

    metadata = await generic._runtime_settings_metadata(
        base_url=pipe.valves.OPENAI_API_BASE_URL,
        api_key=pipe.valves.OPENAI_API_KEY,
        timeout=pipe.valves.OPENAI_API_TIMEOUT,
        model_id="lgos-a/simple-graph",
        metadata={
            "chat_variables": {
                "use_history": False,
                "audience": "expert",
                "stale": "ignored",
            }
        },
    )

    assert metadata == {
        "langgraph_runtime_settings": '{"audience":"expert"}',
    }
    client_factory.assert_called_once_with(
        base_url=pipe.valves.OPENAI_API_BASE_URL,
        api_key=pipe.valves.OPENAI_API_KEY,
        timeout=pipe.valves.OPENAI_API_TIMEOUT,
    )
    client.models.retrieve.assert_awaited_once_with(
        model="simple-graph",
        extra_headers={"x-model-provider": "lgos-a"},
    )
    client.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_passes_runtime_settings_to_initial_and_resume_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    pipe.valves.OPENAI_API_TIMEOUT = 45
    chat = ScriptedChat(
        ((), _interrupt_response()),
        (("Approved.",), _completion("Approved.")),
    )
    runtime_metadata = {
        "langgraph_runtime_settings": '{"use_history":true}',
    }
    settings_metadata = AsyncMock(return_value=runtime_metadata)

    async def confirm(_: dict[str, Any]) -> bool:
        return True

    monkeypatch.setattr(generic, "_chat", chat)
    monkeypatch.setattr(generic, "_runtime_settings_metadata", settings_metadata)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body(USER_REQUEST),
            __event_call__=confirm,
            __metadata__={
                "chat_id": "chat-1",
                "chat_variables": {"use_history": True},
            },
        )
    )

    assert chunks == ["Approved."]
    assert chat.runtime_metadata_calls == [runtime_metadata, runtime_metadata]
    assert chat.timeouts == [45, 45]
    settings_metadata.assert_awaited_once_with(
        base_url=pipe.valves.OPENAI_API_BASE_URL,
        api_key=pipe.valves.OPENAI_API_KEY,
        timeout=pipe.valves.OPENAI_API_TIMEOUT,
        model_id=MODEL_ID,
        metadata={
            "chat_id": "chat-1",
            "chat_variables": {"use_history": True},
        },
    )


async def test_pipe_streams_markdown_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat((MARKDOWN_DELTAS, _completion(MARKDOWN_RESPONSE)))
    monkeypatch.setattr(generic, "_chat", chat)

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
    monkeypatch.setattr(generic, "_chat", chat)

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


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            {"description": "Generating audio"},
            {
                "description": "Generating audio",
                "done": False,
                "hidden": False,
            },
            id="defaults",
        ),
        pytest.param(
            {
                "description": "Audio ready",
                "done": True,
                "hidden": True,
            },
            {
                "description": "Audio ready",
                "done": True,
                "hidden": True,
            },
            id="explicit",
        ),
    ],
)
def test_pipe_maps_status_event_to_openwebui_status(
    data: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "langgraph_openai_serve": {
                "schema_version": 1,
                "event": {
                    "type": "status",
                    "namespace": [],
                    "data": data,
                },
            },
        }
    )

    assert generic._status_event(chunk) == {
        "type": "status",
        "data": expected,
    }


async def test_content_stream_emits_status_event() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "langgraph_openai_serve": {
                "schema_version": 1,
                "event": {
                    "type": "status",
                    "namespace": [],
                    "data": {
                        "description": "Calculating embeddings",
                        "done": False,
                        "hidden": False,
                    },
                },
            },
        }
    )

    async def stream():
        yield ChunkEvent(
            type="chunk",
            chunk=chunk,
            snapshot={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL_ID,
                "choices": [],
            },
        )

    emitter = AsyncMock()
    deltas = [delta async for delta in generic._content_deltas(stream(), emitter)]

    assert deltas == []
    emitter.assert_awaited_once_with(
        {
            "type": "status",
            "data": {
                "description": "Calculating embeddings",
                "done": False,
                "hidden": False,
            },
        }
    )
