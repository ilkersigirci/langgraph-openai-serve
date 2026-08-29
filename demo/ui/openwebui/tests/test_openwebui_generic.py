from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.lib.streaming.chat import ChunkEvent
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
        default_headers={"User-Agent": "lgos-openwebui"},
    )
    client.models.list.assert_awaited_once_with()
    client.__aexit__.assert_awaited_once_with(None, None, None)


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

    models = await pipe.pipes()

    assert models == [
        {
            "id": "lgos-future/graph-a",
            "name": "Generic / lgos-future/graph-a",
        },
    ]


@pytest.mark.parametrize(
    (
        "model_id",
        "model_details",
        "metadata",
        "user",
        "expected_request",
    ),
    [
        pytest.param(
            "lgos-a/interruptible-approval",
            model(),
            None,
            None,
            {
                "model": "interruptible-approval",
                "extra_headers": {"x-model-provider": "lgos-a"},
                "messages": [{"role": "user", "content": "hello"}],
            },
            id="minimal",
        ),
        pytest.param(
            "lgos-a/namespace/graph.with.dots",
            model(
                features=["client_events"],
                client_settings={
                    "schema_version": 1,
                    "defaults": {
                        "use_history": False,
                        "audience": "general",
                    },
                },
            ),
            {
                "chat_id": "chat-123",
                "chat_variables": {
                    "use_history": False,
                    "audience": "expert",
                    "stale": "ignored",
                },
            },
            {"id": "user-123"},
            {
                "model": "namespace/graph.with.dots",
                "extra_headers": {"x-model-provider": "lgos-a"},
                "messages": [{"role": "user", "content": "hello"}],
                "user": "user-123",
                "metadata": {
                    "langgraph_stream_events": "v1",
                    "langgraph_runtime_settings": '{"audience":"expert"}',
                    "session_id": "chat-123",
                },
            },
            id="ephemeral-options",
        ),
    ],
)
async def test_pipe_builds_the_openai_stream_request_from_public_inputs(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    model_details: SimpleNamespace,
    metadata: dict[str, Any] | None,
    user: dict[str, Any] | None,
    expected_request: dict[str, Any],
) -> None:
    stream_context = AsyncMock()
    stream_context.__aenter__.return_value = ScriptedStream(
        ("ok",),
        completion("ok"),
    )
    stream_factory = Mock(return_value=stream_context)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = model_details
    client.chat.completions.stream = stream_factory
    monkeypatch.setattr(generic, "AsyncOpenAI", Mock(return_value=client))

    chunks = await collect_response(
        Pipe().pipe(
            body=body("hello", model=f"generic.{model_id}"),
            __metadata__=metadata,
            __user__=user,
        )
    )

    assert chunks == ["ok"]
    stream_factory.assert_called_once_with(**expected_request)
    stream_context.__aexit__.assert_awaited_once_with(None, None, None)
    client.__aexit__.assert_awaited_once_with(None, None, None)


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

    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunk for chunk in chunks if isinstance(chunk, str)) == (
        MARKDOWN_RESPONSE
    )


def _client_event_chunk(
    event_type: str,
    data: dict[str, Any],
    *,
    namespace: list[str] | None = None,
) -> ChunkEvent:
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
                    "type": event_type,
                    "namespace": namespace or [],
                    "data": data,
                },
            },
        }
    )
    return ChunkEvent(
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


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(True, id="streaming"),
        pytest.param(False, id="non-streaming"),
    ],
)
async def test_pipe_honors_stream_mode_and_emits_native_sources(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    stream: bool,
) -> None:
    expected_completion = citation_response()
    chat = ScriptedChat(((MARKDOWN_RESPONSE,), expected_completion))
    monkeypatch.setattr(generic, "_chat", chat)
    complete = AsyncMock(return_value=expected_completion)
    monkeypatch.setattr(generic, "_chat_completion", complete)
    emitter = AsyncMock()

    chunks = await collect_response(
        configured_pipe.pipe(
            body=body(
                "Cite this",
                model="generic.lgos-a/citation-events",
                stream=stream,
            ),
            __event_emitter__=emitter,
        )
    )

    if stream:
        assert chunks == [MARKDOWN_RESPONSE]
        assert len(chat.calls) == 1
        complete.assert_not_awaited()
    else:
        assert chunks == [
            expected_completion.model_dump(mode="json", exclude_none=True)
        ]
        assert chat.calls == []
        complete.assert_awaited_once()
    emitter.assert_awaited_once_with(
        {
            "type": "source",
            "data": {
                "source": {
                    "name": "Example source",
                    "url": "https://example.com/source",
                },
                "document": ["source"],
                "metadata": [
                    {
                        "source": "https://example.com/source",
                        "name": "Example source",
                        "url": "https://example.com/source",
                    }
                ],
            },
        }
    )


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
async def test_pipe_emits_status_event(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
    data: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    chat = ScriptedChat(((_client_event_chunk("status", data),), completion()))
    monkeypatch.setattr(generic, "_chat", chat)
    emitter = AsyncMock()

    chunks = await collect_response(
        configured_pipe.pipe(body=body("hello"), __event_emitter__=emitter)
    )

    assert chunks == []
    emitter.assert_awaited_once_with(
        {
            "type": "status",
            "data": expected,
        }
    )


async def test_pipe_emits_persistent_plot_embed(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    event = _client_event_chunk(
        "artifact",
        {
            "schema_version": 1,
            "id": "revenue",
            "kind": "plotly",
            "title": "Quarterly <revenue>",
            "summary": "Q4 is highest.",
            "figure": {
                "data": [
                    {
                        "type": "bar",
                        "name": "Revenue",
                        "x": ["Q1", "Q2"],
                        "y": [1, 2],
                        "showlegend": False,
                    }
                ],
                "layout": {"title": {"text": "</script>"}},
            },
        },
        namespace=["plots"],
    )
    monkeypatch.setattr(
        generic,
        "_chat",
        ScriptedChat(((event,), completion())),
    )
    emitter = AsyncMock()

    chunks = await collect_response(
        configured_pipe.pipe(body=body("hello"), __event_emitter__=emitter)
    )

    assert chunks == []
    event = emitter.await_args.args[0]
    assert event["type"] == "embeds"
    assert event["data"].keys() == {"embeds"}
    html = event["data"]["embeds"][0]
    assert "Quarterly &lt;revenue&gt;" in html
    assert "https://cdn.plot.ly/plotly-3.6.0.min.js" in html
    assert '"showlegend":false' in html
    assert "\\u003c/script>" in html
    assert 'Plotly.newPlot("plot", figure.data, figure.layout' in html
