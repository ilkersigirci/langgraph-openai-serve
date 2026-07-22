import importlib
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from openai.types import Model

from lgos_chainlit.lgos_protocol import ModelClientSettings

EXPECTED_WIDGET_COUNT = 3


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


class Session:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values if values is not None else {"chat_profile": "simple"}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def chat_settings_spy(monkeypatch: pytest.MonkeyPatch, simple):
    form = Mock(send=AsyncMock(), refresh=AsyncMock())
    factory = Mock(return_value=form)
    monkeypatch.setattr(simple.cl, "ChatSettings", factory)
    return factory, form


@pytest.mark.anyio
async def test_selected_model_is_retrieved_before_chat_settings_are_sent(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {
                "use_history": False,
                "mode": "detailed",
                "assistant_name": "Guide",
            },
            simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: {"stale": True},
        }
    )

    async def retrieve(_model_id: str) -> Model:
        assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None
        live_settings = session.values["chat_settings"]
        assert isinstance(live_settings, dict)
        live_settings["mode"] = "brief"
        return configured_model(runtime_client_settings)

    retrieve_model = AsyncMock(side_effect=retrieve)
    factory, form = chat_settings_spy(monkeypatch, simple)

    async def send() -> None:
        assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None

    form.send.side_effect = send
    monkeypatch.setattr(simple.discovery_client.models, "retrieve", retrieve_model)
    monkeypatch.setattr(simple.cl, "user_session", session)

    result = await simple.configure_chat_settings()

    assert result is None
    retrieve_model.assert_awaited_once_with("simple")
    widgets = factory.call_args.args[0]
    assert len(widgets) == EXPECTED_WIDGET_COUNT
    assert [widget.initial for widget in widgets] == [False, "detailed", "Guide"]
    form.send.assert_awaited_once_with()
    form.refresh.assert_not_awaited()
    assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] == (
        runtime_client_settings.defaults
    )


@pytest.mark.anyio
async def test_message_handler_sends_changed_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {
                "use_history": False,
                "mode": "brief",
                "assistant_name": "Helper",
            },
            simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: (
                runtime_client_settings.defaults
            ),
        }
    )
    messages = [{"role": "user", "content": "Hello"}]
    stream = MagicMock()
    stream.__aiter__.return_value = iter([])
    stream.close = AsyncMock()
    create = AsyncMock(return_value=stream)
    assistant_message = Mock(content="")
    assistant_message.update = AsyncMock()

    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "Message", Mock(return_value=assistant_message))
    monkeypatch.setattr(simple, "text_only_chat_messages", lambda: messages)
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(simple.client.chat.completions, "create", create)

    await simple.on_message(Mock(content="Hello"))

    create.assert_awaited_once_with(
        model="simple",
        messages=messages,
        stream=True,
        user="demo-user",
        metadata={
            "langgraph_runtime_settings": '{"use_history":false}',
            "langgraph_stream_events": "v1",
        },
    )
    assistant_message.update.assert_awaited_once_with()
    stream.close.assert_awaited_once_with()


@pytest.mark.anyio
async def test_discovery_failure_refreshes_without_committing_settings(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    committed = {"mode": "detailed"}
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": committed,
            simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: (
                runtime_client_settings.defaults
            ),
        }
    )
    factory, form = chat_settings_spy(monkeypatch, simple)
    monkeypatch.setattr(
        simple.discovery_client.models,
        "retrieve",
        AsyncMock(side_effect=RuntimeError("temporarily unavailable")),
    )
    monkeypatch.setattr(simple.cl, "user_session", session)

    await simple.configure_chat_settings()

    factory.assert_called_once_with([])
    form.refresh.assert_awaited_once_with()
    form.send.assert_not_awaited()
    assert session.values["chat_settings"] is committed
    assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


@pytest.mark.anyio
async def test_missing_settings_send_an_empty_committed_form(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "detailed"},
            simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: (
                runtime_client_settings.defaults
            ),
        }
    )
    factory, form = chat_settings_spy(monkeypatch, simple)
    model = Model(id="simple", object="model", created=1, owned_by="proxy")
    monkeypatch.setattr(
        simple.discovery_client.models,
        "retrieve",
        AsyncMock(return_value=model),
    )
    monkeypatch.setattr(simple.cl, "user_session", session)

    await simple.configure_chat_settings()

    factory.assert_called_once_with([])
    form.send.assert_awaited_once_with()
    form.refresh.assert_not_awaited()
    assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


@pytest.mark.anyio
async def test_unrenderable_settings_keep_their_advertised_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session()
    settings = ModelClientSettings(
        schema_version=1,
        json_schema={
            "type": "object",
            "properties": {"temperature": {"type": "number", "default": 0.5}},
        },
        defaults={"temperature": 0.5},
    )
    factory, form = chat_settings_spy(monkeypatch, simple)
    monkeypatch.setattr(
        simple.discovery_client.models,
        "retrieve",
        AsyncMock(return_value=configured_model(settings)),
    )
    monkeypatch.setattr(simple.cl, "user_session", session)

    await simple.configure_chat_settings()

    factory.assert_called_once_with([])
    form.send.assert_awaited_once_with()
    assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] == {
        "temperature": 0.5
    }


@pytest.mark.anyio
async def test_failed_settings_publication_leaves_defaults_inactive(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "detailed"},
            simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: {"stale": True},
        }
    )
    _, form = chat_settings_spy(monkeypatch, simple)
    form.send.side_effect = RuntimeError("socket closed")
    monkeypatch.setattr(
        simple.discovery_client.models,
        "retrieve",
        AsyncMock(return_value=configured_model(runtime_client_settings)),
    )
    monkeypatch.setattr(simple.cl, "user_session", session)

    with pytest.raises(RuntimeError, match="socket closed"):
        await simple.configure_chat_settings()

    assert session.values[simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] is None


@pytest.mark.anyio
async def test_local_transport_errors_are_reported_by_the_message_handler(
    monkeypatch: pytest.MonkeyPatch,
    runtime_client_settings: ModelClientSettings,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "x" * 600},
            simple.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: (
                runtime_client_settings.defaults
            ),
        }
    )

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content
            self.metadata = None

        async def send(self) -> None:
            sent_messages.append(self)

    sent_messages: list[FakeMessage] = []
    create = AsyncMock()
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "Message", FakeMessage)
    monkeypatch.setattr(
        simple,
        "text_only_chat_messages",
        lambda: [{"role": "user", "content": "Hello"}],
    )
    monkeypatch.setattr(simple.client.chat.completions, "create", create)

    await simple.on_message(FakeMessage("Hello"))

    create.assert_not_awaited()
    assert len(sent_messages) == 1
    assert "exceed the OpenAI metadata value limit" in sent_messages[0].content
