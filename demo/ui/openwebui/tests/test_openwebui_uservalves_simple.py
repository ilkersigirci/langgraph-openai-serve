from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lgos_openwebui.functions import uservalves_simple
from lgos_openwebui.functions.uservalves_simple import Pipe


class Stream:
    async def __aiter__(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
        )


def _configured_model() -> SimpleNamespace:
    return SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "DUMMY",
                "features": [],
                "client_settings": {
                    "schema_version": 1,
                    "json_schema": {"type": "object"},
                    "defaults": {"use_history": False, "audience": "general"},
                },
            }
        }
    )


@pytest.mark.anyio
async def test_uservalves_simple_forwards_only_changed_user_valves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = _configured_model()
    client.chat.completions.create.return_value = Stream()
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(pipe, "_client", client_factory)

    chunks = [
        chunk
        async for chunk in pipe.pipe(
            body={"messages": [{"role": "user", "content": "Hi"}]},
            __user__={"valves": pipe.UserValves(use_history=True)},
        )
    ]

    assert chunks == ["Hello"]
    client.models.retrieve.assert_awaited_once_with(
        model="simple-graph",
        extra_headers={"x-model-provider": "lgos-a"},
    )
    client.chat.completions.create.assert_awaited_once_with(
        model="simple-graph",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=[{"role": "user", "content": "Hi"}],
        metadata={"langgraph_runtime_settings": '{"use_history":true}'},
        stream=True,
    )
    assert pipe._runtime_settings_metadata({"valves": pipe.UserValves()}) == {}


@pytest.mark.anyio
async def test_uservalves_simple_uses_the_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    pipe.valves.MODEL = "other-graph"
    pipe.valves.OPENAI_API_HEADERS = {}
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = _configured_model()
    client.chat.completions.create.return_value = Stream()
    monkeypatch.setattr(pipe, "_client", Mock(return_value=client))

    chunks = [chunk async for chunk in pipe.pipe(body={"messages": []})]

    assert chunks == ["Hello"]
    client.models.retrieve.assert_awaited_once_with(model="other-graph")
    client.chat.completions.create.assert_awaited_once_with(
        model="other-graph",
        messages=[],
        metadata={},
        stream=True,
    )


def test_uservalves_simple_uses_standard_model_ids_without_headers() -> None:
    pipe = Pipe()
    pipe.valves.MODEL = "lgos-b/simple-graph"
    pipe.valves.OPENAI_API_HEADERS = {}

    assert pipe._model_request() == {"model": "lgos-b/simple-graph"}


@pytest.mark.anyio
async def test_uservalves_simple_requires_advertised_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": {
                "schema_version": 1,
                "description": "DUMMY",
                "features": [],
            }
        }
    )
    client.chat.completions.create.return_value = Stream()
    monkeypatch.setattr(pipe, "_client", Mock(return_value=client))

    chunks = [
        chunk
        async for chunk in pipe.pipe(
            body={"messages": []},
            __user__={"valves": Pipe.UserValves(use_history=True)},
        )
    ]

    assert chunks == ["Hello"]
    client.chat.completions.create.assert_awaited_once_with(
        model="simple-graph",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=[],
        metadata={},
        stream=True,
    )


@pytest.mark.anyio
async def test_uservalves_simple_warns_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = Pipe()
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.models.retrieve.return_value = SimpleNamespace(model_extra={})
    client.chat.completions.create.return_value = Stream()
    emitter = AsyncMock()
    monkeypatch.setattr(pipe, "_client", Mock(return_value=client))

    chunks = [
        chunk
        async for chunk in pipe.pipe(
            body={"messages": []},
            __user__={"valves": Pipe.UserValves(use_history=True)},
            __event_emitter__=emitter,
        )
    ]

    assert chunks == ["Hello"]
    emitter.assert_awaited_once_with(
        {
            "type": "notification",
            "data": {
                "type": "warning",
                "content": uservalves_simple.LIMITED_FUNCTIONALITY_MESSAGE,
            },
        }
    )
    client.chat.completions.create.assert_awaited_once_with(
        model="simple-graph",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=[],
        metadata={},
        stream=True,
    )
