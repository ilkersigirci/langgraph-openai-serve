"""Chat-settings behavior of the simple Chainlit application."""

import importlib
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from openai.types import Model

from lgos_chainlit.lgos_protocol import ModelClientSettings


class Session:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def configured_model(settings: ModelClientSettings) -> Model:
    return Model(
        id="simple",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": [],
            "client_settings": settings.model_dump(mode="json"),
        },
    )


def chat_settings_spy(monkeypatch: pytest.MonkeyPatch, chat_settings):
    form = Mock(send=AsyncMock(), refresh=AsyncMock())
    factory = Mock(return_value=form)
    monkeypatch.setattr(chat_settings.cl, "ChatSettings", factory)
    return factory, form


@pytest.mark.anyio
async def test_discovered_settings_are_published(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    chat_settings = importlib.import_module("lgos_chainlit.utils.chat_settings")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {
                "use_history": False,
                "mode": "detailed",
                "assistant_name": "Guide",
            },
        }
    )
    retrieve = AsyncMock(return_value=configured_model(runtime_client_settings))
    factory, form = chat_settings_spy(monkeypatch, chat_settings)
    monkeypatch.setattr(chat_settings.discovery_client.models, "retrieve", retrieve)
    monkeypatch.setattr(chat_settings.cl, "user_session", session)

    await chat_settings.configure_chat_settings()

    retrieve.assert_awaited_once_with("simple")
    assert [
        (type(widget).__name__, widget.id, widget.initial)
        for widget in factory.call_args.args[0]
    ] == [
        ("Switch", "use_history", False),
        ("Select", "mode", "detailed"),
        ("TextInput", "assistant_name", "Guide"),
    ]
    form.send.assert_awaited_once_with()
    assert session.values[chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] == (
        runtime_client_settings.defaults
    )


@pytest.mark.anyio
async def test_discovery_failure_disables_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_settings = importlib.import_module("lgos_chainlit.utils.chat_settings")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "detailed"},
            chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: {"stale": True},
        }
    )
    factory, form = chat_settings_spy(monkeypatch, chat_settings)
    monkeypatch.setattr(
        chat_settings.discovery_client.models,
        "retrieve",
        AsyncMock(side_effect=RuntimeError("temporarily unavailable")),
    )
    monkeypatch.setattr(chat_settings.cl, "user_session", session)

    await chat_settings.configure_chat_settings()

    factory.assert_called_once_with([])
    form.refresh.assert_awaited_once_with()
    assert session.values[chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


@pytest.mark.anyio
async def test_model_without_settings_clears_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_settings = importlib.import_module("lgos_chainlit.utils.chat_settings")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "detailed"},
        }
    )
    factory, form = chat_settings_spy(monkeypatch, chat_settings)
    monkeypatch.setattr(
        chat_settings.discovery_client.models,
        "retrieve",
        AsyncMock(
            return_value=Model(
                id="simple",
                object="model",
                created=1,
                owned_by="proxy",
            )
        ),
    )
    monkeypatch.setattr(chat_settings.cl, "user_session", session)

    await chat_settings.configure_chat_settings()

    factory.assert_called_once_with([])
    form.send.assert_awaited_once_with()
    form.refresh.assert_not_awaited()
    assert session.values[chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


@pytest.mark.anyio
async def test_selected_settings_reach_the_openai_request(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    chat_settings = importlib.import_module("lgos_chainlit.utils.chat_settings")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {
                "use_history": False,
                "mode": "detailed",
                "assistant_name": "Guide",
            },
            chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: (
                runtime_client_settings.defaults
            ),
        }
    )
    messages = [{"role": "user", "content": "Hello"}]
    stream = MagicMock()
    stream.__aiter__.return_value = iter([])
    stream.close = AsyncMock()
    create = AsyncMock(return_value=stream)
    assistant_message = Mock(content="", update=AsyncMock())
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "Message", Mock(return_value=assistant_message))
    monkeypatch.setattr(simple, "text_only_chat_messages", lambda: messages)
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(simple.inference_client.chat.completions, "create", create)

    await simple.on_message(Mock(content="Hello"))

    create.assert_awaited_once_with(
        model="simple",
        messages=messages,
        stream=True,
        user="demo-user",
        metadata={
            "langgraph_runtime_settings": (
                '{"use_history":false,"mode":"detailed","assistant_name":"Guide"}'
            ),
            "langgraph_stream_events": "v1",
        },
    )
