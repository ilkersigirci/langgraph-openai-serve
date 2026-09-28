"""Conversion between LGOS metadata and Chainlit chat settings."""

from openai.types import Model

from lgos_chainlit.lgos_protocol import (
    GraphFeature,
    ModelClientSettings,
    model_client_settings,
    model_supports,
)


def test_model_settings_are_optional(
    runtime_client_settings: ModelClientSettings,
) -> None:
    configured = Model(
        id="simple",
        object="model",
        created=1,
        owned_by="test",
        lgos={
            "description": "DUMMY",
            "features": [],
            "client_settings": runtime_client_settings.model_dump(mode="json"),
        },
    )
    missing = Model(id="proxy", object="model", created=1, owned_by="test")

    assert model_client_settings(configured) == runtime_client_settings
    assert model_client_settings(missing) is None


def test_malformed_settings_keep_the_model_features() -> None:
    model = Model(
        id="simple",
        object="model",
        created=1,
        owned_by="test",
        lgos={
            "description": "DUMMY",
            "features": ["file_inputs"],
            "client_settings": {"json_schema": {}},
        },
    )

    assert model_client_settings(model) is None
    assert model_supports(model, GraphFeature.FILE_INPUTS)
