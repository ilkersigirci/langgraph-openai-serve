"""Verify standalone clients against representative LGOS wire payloads."""

from pathlib import Path
from runpy import run_path

from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    ModelClientSettings,
    ModelDetails,
)
from langgraph_openai_serve.graph.events import (
    client_event,
    client_event_extension,
    status_event,
)
from langgraph_openai_serve.graph.features import GraphFeature

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHAINLIT_PROTOCOL = run_path(
    str(REPOSITORY_ROOT / "demo/ui/chainlit_ui/src/lgos_chainlit/lgos_protocol.py")
)


def test_chainlit_accepts_model_discovery_payload() -> None:
    payload = ModelDetails(
        id="interruptible",
        created=1,
        owned_by="langgraph-openai-serve",
        langgraph_openai_serve=LangGraphModelExtension(
            features=[GraphFeature.INTERRUPTS],
            client_settings=ModelClientSettings(
                json_schema={"type": "object", "additionalProperties": False},
                defaults={},
            ),
        ),
    ).model_dump(mode="json")
    extension = payload[CHAINLIT_PROTOCOL["LGOS_EXTENSION_KEY"]]

    parsed = CHAINLIT_PROTOCOL["LangGraphModelExtension"].model_validate(extension)

    assert parsed.model_dump(mode="json") == extension


def test_chainlit_accepts_client_event_payload() -> None:
    extension = client_event_extension(
        client_event(
            "progress",
            {"completed": 1, "total": 2},
            namespace=("retrieval",),
        )
    )
    assert extension is not None

    parsed = CHAINLIT_PROTOCOL["ClientEventExtension"].model_validate(extension)

    assert parsed.model_dump(mode="json") == extension


def test_chainlit_accepts_status_event_payload() -> None:
    extension = client_event_extension(status_event("Generating audio"))
    assert extension is not None

    parsed = CHAINLIT_PROTOCOL["StatusUpdate"].model_validate(
        extension["event"]["data"]
    )

    assert parsed.model_dump(mode="json") == {
        "description": "Generating audio",
        "done": False,
        "hidden": False,
    }
