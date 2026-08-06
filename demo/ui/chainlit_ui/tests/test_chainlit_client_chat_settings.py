"""Conversion between LGOS metadata and Chainlit chat settings."""

from openai.types import Model

from lgos_chainlit.lgos_protocol import ModelClientSettings, model_client_settings


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
            "description": "DUMMY",
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
            "description": "DUMMY",
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
