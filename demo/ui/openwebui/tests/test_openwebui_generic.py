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

USER_REQUEST = "Refund order ORDER-123"
UPSTREAM_MODEL_ID = "interruptible-approval"
MODEL_ID = f"lgos-a/{UPSTREAM_MODEL_ID}"
RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
STATE_TOKEN = "state-token-1"
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
        self.calls: list[tuple[list[dict[str, Any]], str]] = []
        self.runtime_metadata_calls: list[dict[str, str] | None] = []
        self.include_client_events_calls: list[bool] = []

    @asynccontextmanager
    async def __call__(
        self,
        *,
        client: Any,
        messages: list[dict[str, Any]],
        model_id: str,
        runtime_metadata: dict[str, str] | None = None,
        include_client_events: bool = False,
    ) -> AsyncIterator[ScriptedStream]:
        step_index = len(self.calls)
        self.calls.append((messages, model_id))
        self.runtime_metadata_calls.append(runtime_metadata)
        self.include_client_events_calls.append(include_client_events)
        if step_index >= len(self._steps):
            raise AssertionError(f"Unexpected chat call {step_index + 1}")

        deltas, completion = self._steps[step_index]
        yield ScriptedStream(deltas, completion)


async def _collect_response(
    pipe_response: AsyncIterator[str | dict[str, Any]],
) -> list[str | dict[str, Any]]:
    return [chunk async for chunk in pipe_response]


async def _run_interrupt_pipe(event_call: Any) -> list[str | dict[str, Any]]:
    return await _collect_response(
        Pipe().pipe(
            body=_body(USER_REQUEST),
            __event_call__=event_call,
            __metadata__={"chat_id": "chat-1", "session_id": "session-1"},
        )
    )


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
            "model": UPSTREAM_MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


def _model(*, features: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "DUMMY",
                "features": features or [],
            }
        }
    )


@pytest.fixture(autouse=True)
def configured_model_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        generic,
        "_retrieve_model",
        AsyncMock(return_value=_model()),
    )


def _interrupt_call(
    interrupt_id: str,
    payload: object,
    *,
    arguments: object | None = None,
    state_token: str = STATE_TOKEN,
) -> dict[str, Any]:
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


def _interrupt_response(arguments: object | None = None) -> ChatCompletion:
    return _completion(
        tool_calls=[
            _interrupt_call(
                "interrupt-1",
                INTERRUPT_PAYLOAD,
                arguments=arguments,
            )
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
            SimpleNamespace(
                id="lgos-a/interruptible-approval",
                owned_by="langgraph-openai-serve",
            ),
            SimpleNamespace(
                id="lgos-a/lgos-rag",
                owned_by="langgraph-openai-serve",
            ),
            SimpleNamespace(
                id="lgos-b/interruptible-approval",
                owned_by="langgraph-openai-serve",
            ),
            SimpleNamespace(
                id="lgos-b/lgos-rag",
                owned_by="langgraph-openai-serve",
            ),
        ]
    )
    client_factory = Mock(return_value=client)
    retrieve_model = AsyncMock()
    monkeypatch.setattr(
        "lgos_openwebui.functions.generic.AsyncOpenAI",
        client_factory,
    )
    monkeypatch.setattr(generic, "_retrieve_model", retrieve_model)

    models = await pipe.pipes()

    assert models == [
        {
            "id": "lgos-a/interruptible-approval",
            "name": "Generic / lgos-a/interruptible-approval",
        },
        {"id": "lgos-a/lgos-rag", "name": "Generic / lgos-a/lgos-rag"},
        {
            "id": "lgos-b/interruptible-approval",
            "name": "Generic / lgos-b/interruptible-approval",
        },
        {"id": "lgos-b/lgos-rag", "name": "Generic / lgos-b/lgos-rag"},
    ]
    client_factory.assert_called_once_with(
        base_url="http://lgos-bifrost:8080/v1",
        api_key="DUMMY",
        timeout=45,
        max_retries=0,
    )
    client.models.list.assert_awaited_once_with()
    retrieve_model.assert_not_awaited()
    client.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_preserves_dots_in_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat((("ok",), _completion("ok")))
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("hello", model="generic.lgos-a/graph.v2"),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == ["ok"]
    assert chat.calls == [([{"role": "user", "content": "hello"}], "lgos-a/graph.v2")]
    assert chat.include_client_events_calls == [False]


async def test_pipes_ignore_non_lgos_catalog_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    catalog = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="lgos-future/graph-a",
                owned_by="langgraph-openai-serve",
            ),
            SimpleNamespace(id="openai/gpt-5", owned_by="openai"),
        ]
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.list.return_value = catalog
    monkeypatch.setattr(generic, "AsyncOpenAI", Mock(return_value=client))
    retrieve_model = AsyncMock()
    monkeypatch.setattr(generic, "_retrieve_model", retrieve_model)

    models = await pipe.pipes()

    assert models == [
        {
            "id": "lgos-future/graph-a",
            "name": "Generic / lgos-future/graph-a",
        },
    ]
    retrieve_model.assert_not_awaited()
    assert generic._model_request("lgos-future/graph-a") == {
        "model": "graph-a",
        "extra_headers": {"x-model-provider": "lgos-future"},
    }


async def test_pipe_warns_when_the_endpoint_strips_lgos_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    chat = ScriptedChat((("ok",), _completion("ok")))
    emitter = AsyncMock()
    monkeypatch.setattr(generic, "_chat", chat)
    monkeypatch.setattr(generic, "_retrieve_model", AsyncMock(return_value=None))

    chunks = await _collect_response(
        pipe.pipe(
            body=_body("hello"),
            __event_emitter__=emitter,
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == ["ok"]
    emitter.assert_awaited_once_with(
        {
            "type": "notification",
            "data": {
                "type": "warning",
                "content": generic.LIMITED_FUNCTIONALITY_MESSAGE,
            },
        }
    )


async def test_pipe_requests_client_events_only_when_advertised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = ScriptedChat((("ok",), _completion("ok")))
    monkeypatch.setattr(generic, "_chat", chat)
    monkeypatch.setattr(
        generic,
        "_retrieve_model",
        AsyncMock(return_value=_model(features=["client_events"])),
    )

    chunks = await _collect_response(
        Pipe().pipe(
            body=_body("hello"),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == ["ok"]
    assert chat.include_client_events_calls == [True]


async def test_pipe_rejects_model_without_function_prefix() -> None:
    chunks = await _collect_response(Pipe().pipe(body=_body("hello", model="graph")))

    assert chunks == ["Open WebUI did not provide a valid model ID."]


async def test_pipe_requires_a_provider_qualified_bifrost_model() -> None:
    chunks = await _collect_response(
        Pipe().pipe(body=_body("hello", model="generic.interruptible-approval"))
    )

    assert chunks == [
        "Bifrost model ID must use the provider/model format: 'interruptible-approval'."
    ]


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
    chat = ScriptedChat(
        ((), _interrupt_response()),
        (answer_deltas, _completion("".join(answer_deltas))),
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return approved

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(confirm)

    assert chunks == list(answer_deltas)
    assert events == [
        {
            "type": "confirmation",
            "data": {"title": "Approve?", "message": USER_REQUEST},
        }
    ]
    (
        (initial_messages, initial_model_id),
        (resume_messages, resume_model_id),
    ) = chat.calls
    assert initial_messages == [{"role": "user", "content": USER_REQUEST}]
    assert resume_messages[0]["tool_calls"][0]["id"] == ("lg_interrupt_interrupt-1")
    assert json.loads(resume_messages[0]["tool_calls"][0]["function"]["arguments"]) == {
        "run_id": RUN_ID,
        "state_token": STATE_TOKEN,
        "payload": INTERRUPT_PAYLOAD,
    }
    assert resume_messages[1] == {
        "role": "tool",
        "tool_call_id": "lg_interrupt_interrupt-1",
        "content": json.dumps({"resume": decision}),
    }
    assert (initial_model_id, resume_model_id) == (MODEL_ID, MODEL_ID)


@pytest.mark.parametrize(
    ("payload", "message"),
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
    ],
)
async def test_approval_event_renders_every_json_payload(
    payload: object,
    message: str,
) -> None:
    tool_call = (
        _completion(tool_calls=[_interrupt_call("interrupt-1", payload)])
        .choices[0]
        .message.tool_calls[0]
    )

    assert generic._interrupt_payload(tool_call) == payload
    assert generic._approval_event(tool_call) == {
        "type": "confirmation",
        "data": {
            "title": "Approve this agent action?",
            "message": message,
        },
    }


async def test_approval_event_rejects_a_missing_payload() -> None:
    arguments = {
        "run_id": RUN_ID,
        "state_token": STATE_TOKEN,
    }
    tool_call = (
        _completion(
            tool_calls=[
                _interrupt_call("interrupt-1", {}, arguments=arguments),
            ]
        )
        .choices[0]
        .message.tool_calls[0]
    )

    assert generic._approval_event(tool_call) is None


async def test_pipe_collects_every_interrupt_and_resumes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_call = _interrupt_call(
        "interrupt-1",
        {"question": "Approve refund?", "request": USER_REQUEST},
    )
    second_call = _interrupt_call(
        "interrupt-2",
        {
            "question": "Approve notification?",
            "request": "Email the customer",
        },
    )
    interrupt_response = _completion(tool_calls=[first_call, second_call])
    chat = ScriptedChat(
        ((), interrupt_response),
        (("Applied.",), _completion("Applied.")),
    )
    events: list[dict[str, Any]] = []
    answers = iter([True, False])

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return next(answers)

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(confirm)

    assert chunks == ["Applied."]
    assert events == [
        {
            "type": "confirmation",
            "data": {"title": "Approve refund?", "message": USER_REQUEST},
        },
        {
            "type": "confirmation",
            "data": {
                "title": "Approve notification?",
                "message": "Email the customer",
            },
        },
    ]
    assert len(chat.calls) == 2
    resume_messages = chat.calls[1][0]
    assert resume_messages[0] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [first_call, second_call],
    }
    assert resume_messages[1:] == [
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


async def test_pipe_keeps_the_ledger_across_interrupt_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_call = _interrupt_call("interrupt-1", {"question": "First?"})
    second_call = _interrupt_call(
        "interrupt-2",
        {"question": "Second?"},
        state_token="state-token-2",
    )
    chat = ScriptedChat(
        ((), _completion(tool_calls=[first_call])),
        ((), _completion(tool_calls=[second_call])),
        (("Done.",), _completion("Done.")),
    )
    answers = iter([True, False])

    async def confirm(_: dict[str, Any]) -> bool:
        return next(answers)

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(confirm)

    assert chunks == ["Done."]
    assert len(chat.calls) == 3
    first_ledger = chat.calls[1][0]
    second_ledger = chat.calls[2][0]
    assert [message["role"] for message in first_ledger] == ["assistant", "tool"]
    assert [message["role"] for message in second_ledger] == ["assistant", "tool"]
    assert first_ledger[0]["tool_calls"] == [first_call]
    assert second_ledger[0]["tool_calls"] == [second_call]


async def test_pipe_does_not_partially_resume_cancelled_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = ScriptedChat(
        (
            (),
            _completion(
                tool_calls=[
                    _interrupt_call("interrupt-1", {"question": "First?"}),
                    _interrupt_call("interrupt-2", {"question": "Second?"}),
                ]
            ),
        )
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool | None:
        events.append(event)
        return None

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(confirm)

    assert chunks == ["Open WebUI approval was cancelled or timed out."]
    assert len(events) == 1
    assert len(chat.calls) == 1


async def test_pipe_reports_host_approval_failure_without_resuming_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = ScriptedChat(
        (
            (),
            _completion(
                tool_calls=[
                    _interrupt_call("interrupt-1", {"question": "First?"}),
                    _interrupt_call("interrupt-2", {"question": "Second?"}),
                ]
            ),
        )
    )
    event_call = AsyncMock(
        return_value={
            "error": "Event call timed out. The browser tab may be inactive or closed."
        }
    )
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(event_call)

    assert chunks == [
        "Open WebUI approval failed: Event call timed out. "
        "The browser tab may be inactive or closed."
    ]
    event_call.assert_awaited_once_with(
        {
            "type": "confirmation",
            "data": {
                "title": "First?",
                "message": json.dumps(
                    {"question": "First?"},
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        }
    )
    assert len(chat.calls) == 1


@pytest.mark.parametrize("mixed", [False, True], ids=["ordinary", "mixed"])
async def test_pipe_reports_unsupported_tool_call_batches(
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
) -> None:
    ordinary_call = {
        "id": "call_other",
        "type": "function",
        "function": {"name": "other_tool", "arguments": "{}"},
    }
    tool_calls = [ordinary_call]
    if mixed:
        tool_calls.insert(0, _interrupt_call("interrupt-1", INTERRUPT_PAYLOAD))
    chat = ScriptedChat(((), _completion(tool_calls=tool_calls)))
    event_call = AsyncMock(return_value=True)
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(event_call)

    assert chunks == ["Open WebUI received an unsupported tool-call batch."]
    event_call.assert_not_awaited()
    assert len(chat.calls) == 1


async def test_pipe_does_not_resume_a_malformed_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = ScriptedChat(
        ((), _interrupt_response([])),
    )
    events: list[dict[str, Any]] = []

    async def confirm(event: dict[str, Any]) -> bool:
        events.append(event)
        return True

    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await _run_interrupt_pipe(confirm)

    assert chunks == ["Open WebUI received an unsupported interrupt payload."]
    assert events == []
    assert len(chat.calls) == 1


async def test_chat_omits_metadata_when_no_ephemeral_options_are_needed() -> None:
    messages = [{"role": "user", "content": "hello"}]
    stream = ScriptedStream(("ok",), _completion("ok"))
    stream_context = AsyncMock()
    stream_context.__aenter__.return_value = stream
    client = AsyncMock()
    stream_factory = Mock(return_value=stream_context)
    client.chat.completions.stream = stream_factory

    async with generic._chat(
        client=client,
        messages=messages,
        model_id="lgos-a/interruptible-approval",
    ):
        pass

    stream_factory.assert_called_once_with(
        model="interruptible-approval",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=messages,
    )


async def test_chat_sends_model_and_ephemeral_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    completion = _completion("ok")
    stream = ScriptedStream(("ok",), completion)
    stream_context = AsyncMock()
    stream_context.__aenter__.return_value = stream
    client = AsyncMock()
    stream_factory = Mock(return_value=stream_context)
    client.chat.completions.stream = stream_factory

    async with generic._chat(
        client=client,
        messages=messages,
        model_id="lgos-a/namespace/graph.with.dots",
        runtime_metadata={"langgraph_runtime_settings": '{"mode":"detailed"}'},
        include_client_events=True,
    ) as response_stream:
        deltas = [
            event.delta
            async for event in response_stream
            if isinstance(event, ContentDeltaEvent)
        ]
        response = await response_stream.get_final_completion()

    assert response == completion
    assert deltas == ["ok"]
    stream_factory.assert_called_once_with(
        model="namespace/graph.with.dots",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=messages,
        metadata={
            "langgraph_stream_events": "v1",
            "langgraph_runtime_settings": '{"mode":"detailed"}',
        },
    )
    stream_context.__aexit__.assert_awaited_once_with(None, None, None)


async def test_pipe_forwards_changed_chat_variables_as_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "DUMMY",
                "features": [],
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
    metadata = generic._runtime_settings_metadata(
        model=model,
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
    settings_metadata = Mock(return_value=runtime_metadata)

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
    settings_metadata.assert_called_once_with(
        model=_model(),
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
            body=_body("Cite this", model="generic.lgos-a/lgos-rag"),
            __metadata__={"chat_id": "chat-1"},
        )
    )

    assert chunks == list(MARKDOWN_DELTAS)
    assert chat.calls[0][1:] == ("lgos-a/lgos-rag",)


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
                model="generic.lgos-a/citation-events",
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
    assert chat.calls[0][1:] == ("lgos-a/citation-events",)


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
            "model": UPSTREAM_MODEL_ID,
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
            "model": UPSTREAM_MODEL_ID,
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
                "model": UPSTREAM_MODEL_ID,
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
