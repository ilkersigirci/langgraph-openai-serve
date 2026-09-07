"""Verify standalone clients against representative LGOS wire payloads."""

import json
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest
from openai.types import Model as OpenAIModel

from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    ModelClientSettings,
    ModelDetails,
)
from langgraph_openai_serve.graph.client_settings import (
    ClientSettings,
    client_settings_default_values,
    client_settings_json_schema,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.protocol import (
    CONVERSATION_METADATA_KEY,
    INTERRUPT_TOOL_NAME,
    MODEL_EXTENSION_KEY,
    SETTINGS_METADATA_KEY,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHAINLIT_PROTOCOL = run_path(
    str(REPOSITORY_ROOT / "demo/ui/chainlit_ui/src/lgos_chainlit/lgos_protocol.py")
)
OPENWEBUI_PROTOCOL = run_path(
    str(
        REPOSITORY_ROOT
        / "demo/ui/openwebui/src/lgos_openwebui/functions/generic/contracts.py"
    )
)
OPENWEBUI_WORKSPACE_MODELS = run_path(
    str(REPOSITORY_ROOT / "demo/ui/openwebui/src/lgos_openwebui/workspace_models.py")
)


class ExampleSettings(ClientSettings):
    enabled: bool = True


def _model_payload() -> dict[str, Any]:
    return ModelDetails(
        id="interruptible",
        created=1,
        owned_by="langgraph-openai-serve",
        lgos=LangGraphModelExtension(
            description="DUMMY",
            features=[
                GraphFeature.CLIENT_EVENTS,
                GraphFeature.FILE_INPUTS,
                GraphFeature.INTERRUPTS,
            ],
            client_settings=ModelClientSettings(
                json_schema=client_settings_json_schema(ExampleSettings),
                defaults=client_settings_default_values(ExampleSettings),
            ),
        ),
    ).model_dump(mode="json")


def _openwebui_settings_fields(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    model = OpenAIModel.model_validate(payload)
    return OPENWEBUI_WORKSPACE_MODELS["chat_variable_fields"](model)


def test_chainlit_accepts_model_detail_extension() -> None:
    payload = _model_payload()
    extension = payload[CHAINLIT_PROTOCOL["LGOS_EXTENSION_KEY"]]

    parsed = CHAINLIT_PROTOCOL["LangGraphModelExtension"].model_validate(extension)

    assert parsed.model_dump(mode="json") == extension


@pytest.mark.parametrize(
    "protocol",
    [CHAINLIT_PROTOCOL, OPENWEBUI_PROTOCOL],
    ids=["chainlit", "openwebui"],
)
def test_demo_clients_mirror_lgos_protocol_names(protocol: dict[str, object]) -> None:
    assert protocol["LGOS_EXTENSION_KEY"] == MODEL_EXTENSION_KEY
    assert protocol["CONVERSATION_METADATA_KEY"] == CONVERSATION_METADATA_KEY
    assert protocol["SETTINGS_METADATA_KEY"] == SETTINGS_METADATA_KEY
    assert protocol["INTERRUPT_TOOL_NAME"] == INTERRUPT_TOOL_NAME


def test_server_serializes_the_manifest_model_extension_key() -> None:
    assert MODEL_EXTENSION_KEY in _model_payload()


def test_chainlit_settings_descriptor_ignores_additive_fields() -> None:
    parsed = CHAINLIT_PROTOCOL["ModelClientSettings"].model_validate(
        {
            "schema_version": 1,
            "json_schema": {"type": "object"},
            "defaults": {},
            "future_field": True,
        }
    )

    assert parsed.schema_version == 1


@pytest.mark.parametrize("additive_fields", [False, True], ids=["current", "additive"])
def test_openwebui_accepts_server_settings(additive_fields: bool) -> None:
    payload = _model_payload()
    if additive_fields:
        extension = payload[MODEL_EXTENSION_KEY]
        extension["future_field"] = True
        extension["features"].append("future_feature")
        extension["client_settings"]["future_field"] = True

    assert _openwebui_settings_fields(payload) == (
        {"key": "enabled", "type": "checkbox", "label": "Enabled", "default": True},
    )


def test_openwebui_rejects_unsupported_model_extension_version() -> None:
    payload = _model_payload()
    payload[MODEL_EXTENSION_KEY]["schema_version"] = 2

    assert _openwebui_settings_fields(payload) is None


def test_openwebui_ignores_unsupported_settings_version() -> None:
    payload = _model_payload()
    payload[MODEL_EXTENSION_KEY]["client_settings"]["schema_version"] = 2

    assert _openwebui_settings_fields(payload) == ()


def test_bifrost_graph_providers_allow_native_responses() -> None:
    config_path = REPOSITORY_ROOT / "demo/docker/configs/bifrost/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for provider_name in ("lgos-a", "lgos-b"):
        allowed_requests = config["providers"][provider_name]["custom_provider_config"][
            "allowed_requests"
        ]

        assert allowed_requests["responses"] is True
        assert allowed_requests["responses_stream"] is True
