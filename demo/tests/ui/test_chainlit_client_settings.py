import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from openai.types import Model

from langgraph_openai_serve import GraphFeature
from langgraph_openai_serve.api.models.schemas import ModelClientSettings

EXPECTED_WIDGET_COUNT = 3
CLIENT_SETTINGS_SCHEMA_VERSION = 1


@pytest.fixture(autouse=True)
def chainlit_app_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))


def client_module():
    return importlib.import_module("demo.ui.chainlit_ui.client_settings")


def client_settings() -> ModelClientSettings:
    return ModelClientSettings.model_validate(
        {
            "schema_version": CLIENT_SETTINGS_SCHEMA_VERSION,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "use_history": {
                        "type": "boolean",
                        "title": "Use conversation history",
                        "default": True,
                    },
                    "mode": {
                        "type": "string",
                        "title": "Mode",
                        "enum": ["brief", "detailed"],
                        "default": "brief",
                    },
                    "assistant_name": {
                        "type": "string",
                        "title": "Assistant name",
                        "minLength": 1,
                        "default": "Helper",
                    },
                },
            },
            "defaults": {
                "use_history": True,
                "mode": "brief",
                "assistant_name": "Helper",
            },
        }
    )


def configured_model() -> Model:
    return Model(
        id="simple",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": [],
            "client_settings": client_settings().model_dump(mode="json"),
        },
    )


def test_json_schema_is_converted_to_chainlit_widgets() -> None:
    from chainlit.input_widget import (  # noqa: PLC0415
        Select,
        Switch,
        TextInput,
    )

    widgets, values = client_module().settings_widgets(
        client_settings(),
        {
            "use_history": "invalid",
            "mode": "detailed",
            "assistant_name": "",
        },
    )

    assert [type(widget) for widget in widgets] == [
        Switch,
        Select,
        TextInput,
    ]
    assert values == {
        "use_history": True,
        "mode": "detailed",
        "assistant_name": "Helper",
    }
    assert isinstance(widgets[-1], TextInput)


def test_changed_settings_use_one_metadata_envelope() -> None:
    metadata = client_module().settings_metadata(
        client_settings(),
        {
            "use_history": False,
            "mode": "detailed",
            "assistant_name": "Guide",
        },
    )

    assert metadata == {
        "langgraph_runtime_settings": (
            '{"use_history":false,"mode":"detailed","assistant_name":"Guide"}'
        )
    }


def test_default_settings_are_omitted_from_the_request() -> None:
    metadata = client_module().settings_metadata(
        client_settings(),
        client_settings().defaults,
    )

    assert metadata == {}


def test_live_invalid_values_are_sent_unchanged_for_server_validation() -> None:
    metadata = client_module().settings_metadata(
        client_settings(),
        {"mode": "removed-option"},
    )

    assert metadata == {"langgraph_runtime_settings": '{"mode":"removed-option"}'}


def test_type_invalid_value_equal_to_a_default_is_not_omitted() -> None:
    metadata = client_module().settings_metadata(
        client_settings(),
        {"use_history": 1},
    )

    assert metadata == {"langgraph_runtime_settings": '{"use_history":1}'}


def test_oversized_settings_raise_a_local_transport_error() -> None:
    module = client_module()

    with pytest.raises(
        module.SettingsTransportError,
        match="exceed the OpenAI metadata value limit",
    ):
        module.settings_metadata(
            client_settings(),
            {"mode": "x" * 600},
        )


def test_missing_or_stripped_extension_falls_back_without_settings() -> None:
    model = Model(id="simple", object="model", created=1, owned_by="proxy")
    module = client_module()

    assert module.model_client_settings(model) is None
    assert module.settings_metadata(None, None) == {}


def test_nullable_and_unrenderable_properties_are_not_rendered() -> None:
    settings = ModelClientSettings.model_validate(
        {
            "json_schema": {
                "type": "object",
                "properties": {
                    "nullable": {
                        "anyOf": [{"type": "boolean"}, {"type": "null"}],
                        "default": None,
                    },
                    "number": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                    },
                },
            },
            "defaults": {"nullable": None, "number": 0.5},
        }
    )

    assert client_module().settings_widgets(settings) == ([], {})


def test_string_enum_is_skipped_when_one_option_breaks_its_constraints() -> None:
    settings = ModelClientSettings.model_validate(
        {
            "json_schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["x", "valid"],
                        "minLength": 2,
                        "default": "valid",
                    }
                },
            },
            "defaults": {"mode": "valid"},
        }
    )

    assert client_module().settings_widgets(settings) == ([], {})


def test_unsupported_runtime_settings_do_not_hide_known_features() -> None:
    module = client_module()
    model = Model(
        id="interruptible",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": ["interrupts", "future-feature"],
            "client_settings": {"schema_version": 3},
        },
    )

    assert module.model_supports(model, GraphFeature.INTERRUPTS)
    assert module.model_client_settings(model) is None


class Session:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {"chat_profile": "simple"}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


@pytest.mark.anyio
async def test_selected_model_is_retrieved_before_chat_settings_are_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("demo.ui.chainlit_ui.simple")
    session = Session()

    class ChatSettings:
        def __init__(self, widgets):
            assert len(widgets) == EXPECTED_WIDGET_COUNT

        async def send(self):
            values = {
                "use_history": False,
                "mode": "brief",
                "assistant_name": "Configured in Chainlit",
            }
            session.set("chat_settings", values)
            return values

    retrieve = AsyncMock(return_value=configured_model())
    monkeypatch.setattr(simple.discovery_client.models, "retrieve", retrieve)
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "ChatSettings", ChatSettings)

    values = await simple.configure_chat_settings()

    retrieve.assert_awaited_once_with("simple")
    assert values["use_history"] is False
    assert values["assistant_name"] == "Configured in Chainlit"
    stored_settings = session.values["model_client_settings"]
    assert isinstance(stored_settings, dict)
    assert stored_settings.get("schema_version") == CLIENT_SETTINGS_SCHEMA_VERSION
    assert session.values["chat_settings"] == values


@pytest.mark.anyio
async def test_discovery_failure_refreshes_ui_without_committing_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("demo.ui.chainlit_ui.simple")
    committed = {"mode": "detailed"}
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": committed,
            "model_client_settings": client_settings().model_dump(mode="json"),
        }
    )
    refreshed = False

    class ChatSettings:
        def __init__(self, widgets):
            assert widgets == []

        async def refresh(self):
            nonlocal refreshed
            refreshed = True

        async def send(self):
            pytest.fail("a transient discovery failure must not commit settings")

    monkeypatch.setattr(
        simple.discovery_client.models,
        "retrieve",
        AsyncMock(side_effect=RuntimeError("temporarily unavailable")),
    )
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "ChatSettings", ChatSettings)

    values = await simple.configure_chat_settings()

    assert refreshed is True
    assert values is committed
    assert session.values["chat_settings"] is committed
    assert session.values["model_client_settings"] is None


@pytest.mark.anyio
async def test_definitive_missing_settings_clears_committed_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("demo.ui.chainlit_ui.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "detailed"},
            "model_client_settings": client_settings().model_dump(mode="json"),
        }
    )
    sent = False

    class ChatSettings:
        def __init__(self, widgets):
            assert widgets == []

        async def send(self):
            nonlocal sent
            sent = True
            session.set("chat_settings", {})
            return {}

    model = Model(id="simple", object="model", created=1, owned_by="proxy")
    monkeypatch.setattr(
        simple.discovery_client.models,
        "retrieve",
        AsyncMock(return_value=model),
    )
    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "ChatSettings", ChatSettings)

    values = await simple.configure_chat_settings()

    assert sent is True
    assert values == {}
    assert session.values["chat_settings"] == {}
    assert session.values["model_client_settings"] is None


@pytest.mark.anyio
async def test_local_settings_transport_errors_are_reported_by_the_message_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("demo.ui.chainlit_ui.simple")
    session = Session(
        {
            "chat_profile": "simple",
            "chat_settings": {"mode": "x" * 600},
            "model_client_settings": client_settings().model_dump(mode="json"),
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


def test_invalid_saved_values_are_replaced_only_when_restoring_widgets() -> None:
    assert client_module().restore_setting_values(
        client_settings(),
        {"mode": "removed-option"},
    ) == {
        "use_history": True,
        "mode": "brief",
        "assistant_name": "Helper",
    }
