"""Chat-settings behavior of the simple Chainlit application."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai import OpenAIError
from openai.types import Model
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText

from lgos_chainlit.gateway import gateway_config
from lgos_chainlit.lgos_protocol import ModelClientSettings


class Session:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def completed_response(content: str) -> Response:
    return Response.model_construct(
        status="completed",
        output=[
            ResponseOutputMessage(
                id="msg_final",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        logprobs=[],
                        text=content,
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
                phase="final_answer",
            )
        ],
    )


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
                "lgos_chainlit_stream": False,
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
        ("Switch", "lgos_chainlit_stream", False),
        ("Switch", "use_history", False),
        ("Select", "mode", "detailed"),
        ("TextInput", "assistant_name", "Guide"),
    ]
    form.send.assert_awaited_once_with()
    assert session.values[chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY] == (
        runtime_client_settings.defaults
    )
    assert session.values[chat_settings.MODEL_FEATURES_SESSION_KEY] == []


async def test_chat_profiles_use_list_capabilities_for_file_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    monkeypatch.setattr(
        simple,
        "list_models",
        AsyncMock(
            return_value=[
                Model(
                    id="configured",
                    object="model",
                    created=1,
                    owned_by="test",
                    langgraph_openai_serve={
                        "schema_version": 1,
                        "description": "DUMMY",
                        "features": ["file_inputs"],
                    },
                ),
                model_without_extension("proxy-model"),
            ]
        ),
    )

    profiles = await simple.set_chat_profiles(None)

    assert [profile.name for profile in profiles] == ["configured", "proxy-model"]
    assert [profile.markdown_description for profile in profiles] == [
        "DUMMY",
        simple.LIMITED_FUNCTIONALITY_MESSAGE,
    ]
    assert [
        profile.config_overrides.features.spontaneous_file_upload.enabled
        for profile in profiles
    ] == [True, False]


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

    assert [widget.id for widget in factory.call_args.args[0]] == [
        chat_settings.STREAMING_SETTING_ID
    ]
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

    assert [widget.id for widget in factory.call_args.args[0]] == [
        chat_settings.STREAMING_SETTING_ID
    ]
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
    assert [widget.id for widget in factory.call_args.args[0]] == [
        chat_settings.STREAMING_SETTING_ID
    ]
    form.send.assert_awaited_once_with()
    send_ui_message.assert_awaited_once_with(
        "Response failed: no model profile is selected."
    )


async def test_file_upload_failure_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session({"chat_profile": "lgos-a/file-input"})
    send_ui_message = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "Message", Mock(return_value=Mock(content="")))
    monkeypatch.setattr(simple, "text_only_chat_messages", list)
    monkeypatch.setattr(
        simple,
        "with_response_file_parts",
        AsyncMock(side_effect=RuntimeError("upload unavailable")),
    )
    monkeypatch.setattr(simple, "send_ui_message", send_ui_message)
    monkeypatch.setattr(simple.openai_client.responses, "create", create)

    await simple.on_message(Mock(content="Summarize it."))

    send_ui_message.assert_awaited_once_with("Response failed: upload unavailable")
    create.assert_not_awaited()


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
                chat_settings.STREAMING_SETTING_ID: False,
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
    create = AsyncMock(return_value=completed_response("Complete answer"))
    assistant_message = Mock(content="", send=AsyncMock(), update=AsyncMock())
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
        clients,
        "gateway",
        gateway_config("bifrost", "https://gateway.example"),
    )
    monkeypatch.setattr(simple.openai_client.responses, "create", create)

    await simple.on_message(Mock(content="Hello"))

    create.assert_awaited_once_with(
        model="simple",
        extra_headers={"x-model-provider": "lgos-a"},
        input=messages,
        store=False,
        tools=[simple.DISPLAY_FILE_TOOL],
        user="demo-user",
        metadata={
            "langgraph_runtime_settings": (
                '{"use_history":false,"mode":"detailed","assistant_name":"Guide"}'
            ),
            "session_id": "thread-123",
        },
    )
    assert assistant_message.content == "Complete answer"
    assistant_message.send.assert_awaited_once_with()


async def test_streaming_can_be_disabled_without_forwarding_the_ui_setting(
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
                chat_settings.STREAMING_SETTING_ID: False,
                "mode": "detailed",
            },
            chat_settings.RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY: (
                runtime_client_settings.defaults
            ),
            chat_settings.MODEL_FEATURES_SESSION_KEY: [],
        }
    )
    messages = [{"role": "user", "content": "Hello"}]
    create = AsyncMock(return_value=completed_response("Complete answer"))
    assistant_message = Mock(content="", send=AsyncMock(), update=AsyncMock())
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
        clients,
        "gateway",
        gateway_config("bifrost", "https://gateway.example"),
    )
    monkeypatch.setattr(simple.openai_client.responses, "create", create)

    await simple.on_message(Mock(content="Hello"))

    create.assert_awaited_once_with(
        model="simple",
        extra_headers={"x-model-provider": "lgos-a"},
        input=messages,
        store=False,
        tools=[simple.DISPLAY_FILE_TOOL],
        user="demo-user",
        metadata={
            "langgraph_runtime_settings": '{"mode":"detailed"}',
            "session_id": "thread-123",
        },
    )
    assert assistant_message.content == "Complete answer"
    assistant_message.send.assert_awaited_once_with()
    assistant_message.update.assert_not_awaited()
