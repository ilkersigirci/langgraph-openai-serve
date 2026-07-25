"""Conversion between LGOS metadata and Chainlit chat settings."""

import importlib

import pytest
from openai.types import Model

from lgos_chainlit.lgos_protocol import ModelClientSettings, model_client_settings


def client_module():
    return importlib.import_module("lgos_chainlit.utils.chat_settings")


def test_invalid_saved_values_use_defaults(
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


def test_unsupported_properties_are_skipped() -> None:
    settings = ModelClientSettings(
        schema_version=1,
        json_schema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "assistant_name": {"type": "string"},
            },
        },
        defaults={"temperature": 0.5, "assistant_name": "Helper"},
    )

    widgets = client_module().settings_widgets(settings)

    assert [(widget.id, widget.initial) for widget in widgets] == [
        ("assistant_name", "Helper")
    ]


def test_metadata_is_omitted_without_changes(
    runtime_client_settings: ModelClientSettings,
) -> None:
    module = client_module()

    assert module.settings_metadata(None, None) == {}
    assert (
        module.settings_metadata(
            runtime_client_settings.defaults,
            runtime_client_settings.defaults,
        )
        == {}
    )


@pytest.mark.parametrize(
    ("defaults", "values"),
    [
        ({"mode": "brief"}, {"mode": "x" * 600}),
        ({"temperature": 0.0}, {"temperature": float("nan")}),
    ],
)
def test_unencodable_metadata_is_rejected(
    defaults: dict[str, object],
    values: dict[str, object],
) -> None:
    module = client_module()

    with pytest.raises(module.SettingsTransportError):
        module.settings_metadata(defaults, values)


def test_model_settings_are_versioned_and_optional(
    runtime_client_settings: ModelClientSettings,
) -> None:
    configured = Model(
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
    unsupported = Model(
        id="future",
        object="model",
        created=1,
        owned_by="test",
        langgraph_openai_serve={
            "schema_version": 1,
            "features": [],
            "client_settings": {
                "schema_version": 2,
                "json_schema": {},
                "defaults": {},
            },
        },
    )
    missing = Model(id="proxy", object="model", created=1, owned_by="test")

    assert model_client_settings(configured) == runtime_client_settings
    assert model_client_settings(unsupported) is None
    assert model_client_settings(missing) is None
