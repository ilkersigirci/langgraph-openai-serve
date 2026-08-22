from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.lib.streaming.chat import ChunkEvent, ContentDeltaEvent
from openai.types.chat import ChatCompletionChunk

from lgos_openwebui.functions import generic
from lgos_openwebui.functions.generic import Pipe

from .openwebui_support import (
    MARKDOWN_DELTAS,
    MARKDOWN_RESPONSE,
    UPSTREAM_MODEL_ID,
    ScriptedChat,
    ScriptedStream,
    body,
    citation_response,
    collect_response,
    completion,
    model,
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
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat((("ok",), completion("ok")))
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await collect_response(
        configured_pipe.pipe(
            body=body("hello", model="generic.lgos-a/graph.v2"),
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
    chat = ScriptedChat((("ok",), completion("ok")))
    emitter = AsyncMock()
    monkeypatch.setattr(generic, "_chat", chat)
    monkeypatch.setattr(generic, "_retrieve_model", AsyncMock(return_value=None))

    chunks = await collect_response(
        pipe.pipe(
            body=body("hello"),
            __event_emitter__=emitter,
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
    chat = ScriptedChat((("ok",), completion("ok")))
    monkeypatch.setattr(generic, "_chat", chat)
    monkeypatch.setattr(
        generic,
        "_retrieve_model",
        AsyncMock(return_value=model(features=["client_events"])),
    )

    chunks = await collect_response(
        Pipe().pipe(
            body=body("hello"),
        )
    )

    assert chunks == ["ok"]
    assert chat.include_client_events_calls == [True]


async def test_pipe_rejects_model_without_function_prefix() -> None:
    chunks = await collect_response(Pipe().pipe(body=body("hello", model="graph")))

    assert chunks == ["Open WebUI did not provide a valid model ID."]


async def test_pipe_requires_a_provider_qualified_bifrost_model() -> None:
    chunks = await collect_response(
        Pipe().pipe(body=body("hello", model="generic.interruptible-approval"))
    )

    assert chunks == [
        "Bifrost model ID must use the provider/model format: 'interruptible-approval'."
    ]


async def test_chat_omits_metadata_when_no_ephemeral_options_are_needed() -> None:
    messages = [{"role": "user", "content": "hello"}]
    stream = ScriptedStream(("ok",), completion("ok"))
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


async def test_chat_sends_model_and_ephemeral_request_metadata() -> None:
    messages = [{"role": "user", "content": "hello"}]
    expected_completion = completion("ok")
    stream = ScriptedStream(("ok",), expected_completion)
    stream_context = AsyncMock()
    stream_context.__aenter__.return_value = stream
    client = AsyncMock()
    stream_factory = Mock(return_value=stream_context)
    client.chat.completions.stream = stream_factory

    async with generic._chat(
        client=client,
        messages=messages,
        model_id="lgos-a/namespace/graph.with.dots",
        request_metadata={
            "langgraph_runtime_settings": '{"mode":"detailed"}',
            "session_id": "chat-123",
        },
        include_client_events=True,
    ) as response_stream:
        deltas = [
            event.delta
            async for event in response_stream
            if isinstance(event, ContentDeltaEvent)
        ]
        response = await response_stream.get_final_completion()

    assert response == expected_completion
    assert deltas == ["ok"]
    stream_factory.assert_called_once_with(
        model="namespace/graph.with.dots",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=messages,
        metadata={
            "langgraph_stream_events": "v1",
            "langgraph_runtime_settings": '{"mode":"detailed"}',
            "session_id": "chat-123",
        },
    )
    stream_context.__aexit__.assert_awaited_once_with(None, None, None)


def test_pipe_maps_chat_and_variables_to_request_metadata() -> None:
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
    metadata = generic._request_metadata(
        model=model,
        metadata={
            "chat_id": "chat-123",
            "chat_variables": {
                "use_history": False,
                "audience": "expert",
                "stale": "ignored",
            },
        },
    )

    assert metadata == {
        "langgraph_runtime_settings": '{"audience":"expert"}',
        "session_id": "chat-123",
    }


async def test_pipe_streams_markdown_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    chat = ScriptedChat((MARKDOWN_DELTAS, completion(MARKDOWN_RESPONSE)))
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await collect_response(
        configured_pipe.pipe(
            body=body("Cite this", model="generic.lgos-a/lgos-rag"),
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
    configured_pipe: Pipe,
    stream: bool,
) -> None:
    expected_completion = citation_response()
    chat = ScriptedChat(((MARKDOWN_RESPONSE,), expected_completion))
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await collect_response(
        configured_pipe.pipe(
            body=body(
                "Cite this",
                model="generic.lgos-a/citation-events",
                stream=stream,
            ),
        )
    )

    expected: list[str | dict[str, Any]] = [MARKDOWN_RESPONSE]
    if stream:
        annotation = expected_completion.choices[0].message.annotations[0]
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
