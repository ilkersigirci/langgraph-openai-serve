"""Verify standalone clients against representative LGOS wire payloads."""

import json
from pathlib import Path
from runpy import run_path

from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    ModelClientSettings,
    ModelDetails,
)
from langgraph_openai_serve.graph.features import GraphFeature

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHAINLIT_PROTOCOL = run_path(
    str(REPOSITORY_ROOT / "demo/ui/chainlit_ui/src/lgos_chainlit/lgos_protocol.py")
)


def test_chainlit_accepts_model_detail_extension() -> None:
    payload = ModelDetails(
        id="interruptible",
        created=1,
        owned_by="langgraph-openai-serve",
        langgraph_openai_serve=LangGraphModelExtension(
            description="DUMMY",
            features=[
                GraphFeature.CLIENT_EVENTS,
                GraphFeature.FILE_INPUTS,
                GraphFeature.INTERRUPTS,
            ],
            client_settings=ModelClientSettings(
                json_schema={"type": "object", "additionalProperties": False},
                defaults={},
            ),
        ),
    ).model_dump(mode="json")
    extension = payload[CHAINLIT_PROTOCOL["LGOS_EXTENSION_KEY"]]

    parsed = CHAINLIT_PROTOCOL["LangGraphModelExtension"].model_validate(extension)

    assert parsed.model_dump(mode="json") == extension


def test_bifrost_graph_providers_allow_native_responses() -> None:
    config_path = REPOSITORY_ROOT / "demo/docker/configs/bifrost/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for provider_name in ("lgos-a", "lgos-b"):
        allowed_requests = config["providers"][provider_name]["custom_provider_config"][
            "allowed_requests"
        ]

        assert allowed_requests["responses"] is True
        assert allowed_requests["responses_stream"] is True
