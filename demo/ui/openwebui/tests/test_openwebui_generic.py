from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
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
    stream_chunk,
)

ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "Ask the user a question.",
        "parameters": {"type": "object", "properties": {}},
    },
}


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
            SimpleNamespace(id="openai/gpt-5", owned_by="openai"),
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


@pytest.mark.parametrize(
    (
        "model_id",
        "model_details",
        "metadata",
        "user",
        "request_options",
        "expected_request",
    ),
    [
        pytest.param(
            "lgos-a/interruptible-approval",
            model(),
            None,
            None,
            {},
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
                "temperature": 0.2,
                "top_p": 0.9,
                "n": 1,
                "stop": ["DONE"],
                "max_tokens": 100,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.3,
                "logit_bias": {"42": -1},
                "tools": [ASK_USER_TOOL],
                "tool_choice": "auto",
                "stream_options": {"include_usage": True},
                "response_format": {"type": "json_object"},
            },
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
                "temperature": 0.2,
                "top_p": 0.9,
                "n": 1,
                "stop": ["DONE"],
                "max_tokens": 100,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.3,
                "logit_bias": {"42": -1},
                "tools": [ASK_USER_TOOL],
                "tool_choice": "auto",
                "stream_options": {"include_usage": True},
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
    request_options: dict[str, Any],
    expected_request: dict[str, Any],
) -> None:
    stream = ScriptedStream(
        ("ok",),
        completion("ok"),
    )
    stream.close = AsyncMock()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = model_details
    client.chat.completions.create.return_value = stream
    monkeypatch.setattr(generic, "AsyncOpenAI", Mock(return_value=client))

    chunks = await collect_response(
        Pipe().pipe(
            body={
                **body("hello", model=f"generic.{model_id}"),
                **request_options,
            },
            __metadata__=metadata,
            __user__=user,
        )
    )

    assert chunks == [
        stream_chunk(content="ok").model_dump(mode="json", exclude_none=True)
    ]
    client.chat.completions.create.assert_awaited_once_with(
        **expected_request,
        stream=True,
    )
    stream.close.assert_awaited_once()
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

    assert chunks == [
        stream_chunk(content="ok").model_dump(mode="json", exclude_none=True)
    ]
    emitter.assert_awaited_once_with(
        {
            "type": "notification",
            "data": {
                "type": "warning",
                "content": generic.LIMITED_FUNCTIONALITY_MESSAGE,
            },
        }
    )


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("graph", id="missing-function-prefix"),
        pytest.param(None, id="not-a-string"),
    ],
)
async def test_pipe_rejects_invalid_model_id(model_id: object) -> None:
    chunks = await collect_response(
        Pipe().pipe(body={**body("hello"), "model": model_id})
    )

    assert chunks == [
        {"error": {"detail": "Open WebUI did not provide a valid model ID."}}
    ]


async def test_pipe_requires_a_provider_qualified_bifrost_model() -> None:
    chunks = await collect_response(
        Pipe().pipe(body=body("hello", model="generic.interruptible-approval"))
    )

    assert chunks == [
        {
            "error": {
                "detail": "Bifrost model ID must use the provider/model format: "
                "'interruptible-approval'."
            }
        }
    ]


async def test_pipe_preserves_standard_stream_chunks(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    content = [stream_chunk(content=delta) for delta in MARKDOWN_DELTAS]
    finish = stream_chunk(finish_reason="stop")
    usage = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": UPSTREAM_MODEL_ID,
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    )
    expected = [*content, finish, usage]
    chat = ScriptedChat((expected, completion()))
    monkeypatch.setattr(generic, "_chat", chat)

    chunks = await collect_response(
        configured_pipe.pipe(
            body=body("Cite this", model="generic.lgos-a/lgos-rag"),
        )
    )

    assert chunks == [
        chunk.model_dump(mode="json", exclude_none=True) for chunk in expected
    ]


def _client_event_chunk(
    event_type: str,
    data: dict[str, Any],
    *,
    namespace: list[str] | None = None,
) -> ChatCompletionChunk:
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
    return chunk


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(True, id="streaming"),
        pytest.param(False, id="non-streaming"),
    ],
)
async def test_pipe_honors_stream_mode_and_preserves_native_annotations(
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
        assert chunks[0] == stream_chunk(content=MARKDOWN_RESPONSE).model_dump(
            mode="json",
            exclude_none=True,
        )
        annotation_chunk = chunks[1]
        assert isinstance(annotation_chunk, dict)
        assert annotation_chunk["choices"][0]["finish_reason"] == "stop"
        assert annotation_chunk["choices"][0]["delta"]["annotations"] == [
            annotation.model_dump(mode="json", exclude_none=True)
            for annotation in expected_completion.choices[0].message.annotations or []
        ]
        assert len(chat.calls) == 1
        complete.assert_not_awaited()
    else:
        assert chunks == [
            expected_completion.model_dump(mode="json", exclude_none=True)
        ]
        assert chat.calls == []
        complete.assert_awaited_once()
    emitter.assert_not_awaited()


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


async def test_pipe_emits_persistent_plot_agent_embed(
    monkeypatch: pytest.MonkeyPatch,
    configured_pipe: Pipe,
) -> None:
    event = _client_event_chunk(
        "artifact",
        {
            "schema_version": 1,
            "id": "revenue",
            "kind": "chart",
            "title": "Quarterly <revenue></script>",
            "summary": "Q4 is highest.",
            "chart_type": "bar",
            "labels": ["Q1", "Q2"],
            "series": [{"name": "Revenue", "values": [1, 2]}],
            "x_axis_title": "Quarter",
            "y_axis_title": "Revenue (USD, thousands)",
            "show_legend": False,
        },
        namespace=["charts"],
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
    await_args = emitter.await_args
    assert await_args is not None
    event = await_args.args[0]
    assert event["type"] == "embeds"
    assert event["data"].keys() == {"embeds", "replace"}
    assert event["data"]["replace"] is True
    html = event["data"]["embeds"][0]
    assert "Quarterly &lt;revenue&gt;" in html
    assert "https://cdn.plot.ly/plotly-4.0.0.min.js" in html
    assert '"showlegend":false' in html
    assert "\\u003c/script>" in html
    assert 'Plotly.newPlot("plot", figure.data, figure.layout' in html
    assert 'parent.postMessage({type: "iframe:height", height}, "*")' in html
