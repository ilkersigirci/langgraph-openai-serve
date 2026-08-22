"""Chat-settings behavior of the simple Chainlit application."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from openai import OpenAIError
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
            "description": "DUMMY",
            "features": [],
            "client_settings": settings.model_dump(mode="json"),
        },
    )


def model_without_extension(model_id: str) -> Model:
    return Model(
        id=model_id,
        object="model",
        created=1,
        owned_by="test",
    )


def chat_settings_spy(monkeypatch: pytest.MonkeyPatch, chat_settings):
    form = Mock(send=AsyncMock(), refresh=AsyncMock())
    factory = Mock(return_value=form)
    monkeypatch.setattr(chat_settings.cl, "ChatSettings", factory)
    return factory, form


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
    monkeypatch.setattr(chat_settings, "retrieve_model", retrieve)
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
    assert session.values[chat_settings.MODEL_FEATURES_SESSION_KEY] == []


async def test_chat_profiles_use_list_only_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    monkeypatch.setattr(
        simple.openai_client.models,
        "list",
        AsyncMock(
            return_value=MagicMock(
                data=[
                    Model(
                        id="configured",
                        object="model",
                        created=1,
                        owned_by="test",
                        langgraph_openai_serve={
                            "schema_version": 1,
                            "description": "DUMMY",
                        },
                    ),
                    model_without_extension("proxy-model"),
                ]
            )
        ),
    )
    retrieve = AsyncMock()
    monkeypatch.setattr(simple.openai_client.models, "retrieve", retrieve)

    profiles = await simple.set_chat_profiles(None)

    assert [profile.name for profile in profiles] == ["configured", "proxy-model"]
    assert [profile.markdown_description for profile in profiles] == [
        "DUMMY",
        simple.LIMITED_FUNCTIONALITY_MESSAGE,
    ]
    retrieve.assert_not_awaited()


async def test_model_retrieval_failure_disables_settings(
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
    warning = AsyncMock()
    monkeypatch.setattr(
        chat_settings,
        "retrieve_model",
        AsyncMock(side_effect=OpenAIError("temporarily unavailable")),
    )
    monkeypatch.setattr(chat_settings.cl, "user_session", session)
    monkeypatch.setattr(
        chat_settings,
        "send_limited_functionality_warning",
        warning,
    )

    await chat_settings.configure_chat_settings()

    factory.assert_called_once_with([])
    form.refresh.assert_awaited_once_with()
    warning.assert_awaited_once_with()
    assert session.values[chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


async def test_model_without_extension_warns_and_clears_settings(
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
    warning = AsyncMock()
    monkeypatch.setattr(
        chat_settings,
        "retrieve_model",
        AsyncMock(return_value=model_without_extension("simple")),
    )
    monkeypatch.setattr(chat_settings.cl, "user_session", session)
    monkeypatch.setattr(
        chat_settings,
        "send_limited_functionality_warning",
        warning,
    )

    await chat_settings.configure_chat_settings()

    factory.assert_called_once_with([])
    form.send.assert_awaited_once_with()
    form.refresh.assert_not_awaited()
    warning.assert_awaited_once_with()
    assert session.values[chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


async def test_missing_profile_disables_settings_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    chat_settings = importlib.import_module("lgos_chainlit.utils.chat_settings")
    session = Session({})
    retrieve = AsyncMock()
    send_ui_message = AsyncMock()
    factory, form = chat_settings_spy(monkeypatch, chat_settings)
    monkeypatch.setattr(chat_settings, "retrieve_model", retrieve)
    monkeypatch.setattr(chat_settings.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple, "send_ui_message", send_ui_message)

    await chat_settings.configure_chat_settings()
    await simple.on_message(Mock())

    retrieve.assert_not_awaited()
    factory.assert_called_once_with([])
    form.send.assert_awaited_once_with()
    send_ui_message.assert_awaited_once_with(
        "Chat completion failed: no model profile is selected."
    )


async def test_selected_settings_reach_the_openai_request(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    clients = importlib.import_module("lgos_chainlit.utils.clients")
    chat_settings = importlib.import_module("lgos_chainlit.utils.chat_settings")
    session = Session(
        {
            "chat_profile": "lgos-a/simple",
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
    monkeypatch.setattr(
        simple.cl,
        "context",
        SimpleNamespace(session=SimpleNamespace(thread_id="thread-123")),
    )
    monkeypatch.setattr(
        clients.settings.OPENAI,
        "catalog_base_url",
        "https://gateway.example/v1",
    )
    monkeypatch.setattr(simple.openai_client.chat.completions, "create", create)

    await simple.on_message(Mock(content="Hello"))

    create.assert_awaited_once_with(
        model="simple",
        extra_headers={"x-model-provider": "lgos-a"},
        messages=messages,
        stream=True,
        user="demo-user",
        metadata={
            "langgraph_runtime_settings": (
                '{"use_history":false,"mode":"detailed","assistant_name":"Guide"}'
            ),
            "session_id": "thread-123",
        },
    )
