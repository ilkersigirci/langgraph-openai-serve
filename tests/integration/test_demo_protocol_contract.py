"""Verify standalone clients against representative LGOS wire payloads."""

import json
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest
from demo.ui.openwebui.src.lgos_openwebui.workspace_models import chat_variable_fields
from openai.types import Model as OpenAIModel

from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    ModelClientSettings,
    ModelDetails,
)
from langgraph_openai_serve.graph.client_settings import ClientSettings
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
                GraphFeature.FILE_INPUTS,
                GraphFeature.INTERRUPTS,
                GraphFeature.MCP_TOOLS,
            ],
            client_settings=ModelClientSettings(
                json_schema=ExampleSettings.json_schema(),
                defaults=ExampleSettings.default_values(),
            ),
        ),
    ).model_dump(mode="json")


def _openwebui_settings_fields(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    model = OpenAIModel.model_validate(payload)
    return chat_variable_fields(model)


def test_chainlit_accepts_model_detail_extension() -> None:
    payload = _model_payload()
    extension = payload[CHAINLIT_PROTOCOL["LGOS_EXTENSION_KEY"]]

    parsed = CHAINLIT_PROTOCOL["LangGraphModelExtension"].model_validate(extension)

    assert parsed.model_dump(mode="json") == extension


def test_chainlit_mirrors_lgos_feature_names() -> None:
    assert (
        CHAINLIT_PROTOCOL["BACKGROUND_FEATURE"],
        CHAINLIT_PROTOCOL["FILE_INPUTS_FEATURE"],
        CHAINLIT_PROTOCOL["MCP_TOOLS_FEATURE"],
    ) == (GraphFeature.BACKGROUND, GraphFeature.FILE_INPUTS, GraphFeature.MCP_TOOLS)


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


def test_bifrost_graph_providers_allow_native_responses() -> None:
    config_path = REPOSITORY_ROOT / "demo/docker/configs/bifrost/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for provider_name in ("lgos-a", "lgos-b"):
        allowed_requests = config["providers"][provider_name]["custom_provider_config"][
            "allowed_requests"
        ]

        assert allowed_requests["responses"] is True
        assert allowed_requests["responses_stream"] is True
