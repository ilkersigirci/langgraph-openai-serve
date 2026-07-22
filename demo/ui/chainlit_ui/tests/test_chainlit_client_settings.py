import importlib

import pytest
from openai.types import Model
from pydantic import ValidationError

from lgos_chainlit.lgos_protocol import GraphFeature, ModelClientSettings


def client_module():
    return importlib.import_module("lgos_chainlit.client_settings")


def protocol_module():
    return importlib.import_module("lgos_chainlit.lgos_protocol")


def scalar_settings(schema: object, default: object) -> ModelClientSettings:
    return ModelClientSettings.model_validate(
        {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "properties": {"value": schema},
            },
            "defaults": {"value": default},
        }
    )


def test_json_schema_is_converted_to_chainlit_widgets(
    runtime_client_settings: ModelClientSettings,
) -> None:
    from chainlit.input_widget import (
        Select,
        Switch,
        TextInput,
    )

    widgets = client_module().settings_widgets(
        runtime_client_settings,
        {
            "use_history": False,
            "mode": "detailed",
            "assistant_name": "Guide",
        },
    )

    assert [type(widget) for widget in widgets] == [
        Switch,
        Select,
        TextInput,
    ]
    assert [widget.initial for widget in widgets] == [False, "detailed", "Guide"]


def test_invalid_widget_values_fall_back_to_advertised_defaults(
    runtime_client_settings: ModelClientSettings,
) -> None:
    widgets = client_module().settings_widgets(
        runtime_client_settings,
        {
            "use_history": 1,
            "mode": "removed-option",
            "assistant_name": False,
        },
    )

    assert [widget.initial for widget in widgets] == [True, "brief", "Helper"]


def test_server_only_string_constraints_are_not_evaluated() -> None:
    settings = scalar_settings(
        {
            "type": "string",
            "pattern": "[",
            "minLength": True,
            "format": "email",
        },
        "Helper",
    )

    widgets = client_module().settings_widgets(settings, {"value": "not-an-email"})

    assert [widget.initial for widget in widgets] == ["not-an-email"]


@pytest.mark.parametrize(
    ("schema", "default"),
    [
        (None, "same"),
        ({}, "same"),
        ({"type": "number"}, 1.0),
        ({"type": ["string", "null"]}, "same"),
        ({"type": "boolean"}, "not-boolean"),
        ({"type": "string"}, True),
        ({"type": "string", "enum": None}, "same"),
        ({"type": "string", "enum": "same"}, "same"),
        ({"type": "string", "enum": []}, "same"),
        ({"type": "string", "enum": ["same", 1]}, "same"),
        ({"type": "string", "enum": ["same", "same"]}, "same"),
        ({"type": "string", "enum": ["other"]}, "same"),
        ({"$ref": "#/$defs/value"}, "same"),
    ],
)
def test_unsupported_widget_shapes_are_skipped_without_raising(
    schema: object,
    default: object,
) -> None:
    settings = scalar_settings(schema, default)

    assert client_module().settings_widgets(settings) == []


def test_unsupported_property_does_not_hide_later_supported_widgets() -> None:
    settings = ModelClientSettings.model_validate(
        {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "assistant_name": {"type": "string"},
                },
            },
            "defaults": {"temperature": 0.5, "assistant_name": "Helper"},
        }
    )

    widgets = client_module().settings_widgets(settings)

    assert [widget.id for widget in widgets] == ["assistant_name"]


def test_empty_property_names_are_skipped() -> None:
    settings = ModelClientSettings.model_validate(
        {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "properties": {"": {"type": "string"}},
            },
            "defaults": {"": "same"},
        }
    )

    assert client_module().settings_widgets(settings) == []


def test_changed_settings_use_one_metadata_envelope(
    runtime_client_settings: ModelClientSettings,
) -> None:
    metadata = client_module().settings_metadata(
        runtime_client_settings.defaults,
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


def test_default_settings_are_omitted_from_the_request(
    runtime_client_settings: ModelClientSettings,
) -> None:
    metadata = client_module().settings_metadata(
        runtime_client_settings.defaults,
        runtime_client_settings.defaults,
    )

    assert metadata == {}


def test_live_invalid_values_are_sent_unchanged_for_server_validation(
    runtime_client_settings: ModelClientSettings,
) -> None:
    metadata = client_module().settings_metadata(
        runtime_client_settings.defaults,
        {"mode": "removed-option"},
    )

    assert metadata == {"langgraph_runtime_settings": '{"mode":"removed-option"}'}


def test_type_invalid_value_equal_to_a_default_is_not_omitted(
    runtime_client_settings: ModelClientSettings,
) -> None:
    metadata = client_module().settings_metadata(
        runtime_client_settings.defaults,
        {"use_history": 1},
    )

    assert metadata == {"langgraph_runtime_settings": '{"use_history":1}'}


def test_oversized_settings_raise_a_local_transport_error(
    runtime_client_settings: ModelClientSettings,
) -> None:
    module = client_module()

    with pytest.raises(
        module.SettingsTransportError,
        match="exceed the OpenAI metadata value limit",
    ):
        module.settings_metadata(
            runtime_client_settings.defaults,
            {"mode": "x" * 600},
        )


def test_non_finite_settings_raise_a_local_transport_error() -> None:
    module = client_module()

    with pytest.raises(
        module.SettingsTransportError,
        match="cannot be encoded as JSON",
    ):
        module.settings_metadata(
            {"temperature": 0.0},
            {"temperature": float("nan")},
        )


def test_missing_or_stripped_extension_falls_back_without_settings() -> None:
    model = Model(id="simple", object="model", created=1, owned_by="proxy")

    assert protocol_module().model_client_settings(model) is None
    assert client_module().settings_metadata(None, None) == {}


def test_runtime_settings_version_is_required() -> None:
    with pytest.raises(ValidationError):
        ModelClientSettings.model_validate(
            {
                "json_schema": {"type": "object"},
                "defaults": {},
            }
        )


def test_model_extension_version_is_required() -> None:
    model = Model(
        id="simple",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={"features": []},
    )

    assert protocol_module().model_extension(model) is None


def test_supported_runtime_settings_are_parsed(
    runtime_client_settings: ModelClientSettings,
) -> None:
    model = Model(
        id="simple",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": [],
            "client_settings": runtime_client_settings.model_dump(mode="json"),
        },
    )

    assert protocol_module().model_client_settings(model) == runtime_client_settings


@pytest.mark.parametrize(
    "raw_settings",
    [
        {"json_schema": {"type": "object"}, "defaults": {}},
        {"schema_version": 3, "json_schema": {}, "defaults": {}},
        "invalid",
        [],
    ],
)
def test_unsupported_runtime_settings_do_not_hide_known_features(
    raw_settings: object,
) -> None:
    module = protocol_module()
    model = Model(
        id="interruptible",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": ["interrupts", "future-feature"],
            "client_settings": raw_settings,
        },
    )

    assert module.model_supports(model, GraphFeature.INTERRUPTS)
    assert module.model_client_settings(model) is None
