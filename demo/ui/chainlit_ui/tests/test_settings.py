import pytest
from pydantic import ValidationError

from lgos_chainlit.settings import Settings


def test_ui_file_rejects_unknown_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_UI_FILE", "../other")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_proxy_discovery_and_inference_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DEMO_CHAINLIT_INFERENCE__BASE_URL", "https://gateway.example/v1"
    )
    monkeypatch.setenv("DEMO_CHAINLIT_INFERENCE__API_KEY", "inference-key")
    monkeypatch.setenv("DEMO_CHAINLIT_DISCOVERY__BASE_URL", "https://lgos.example/v1")
    monkeypatch.setenv("DEMO_CHAINLIT_DISCOVERY__API_KEY", "discovery-key")
    monkeypatch.setenv("DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX", "lgos/")

    configured = Settings(_env_file=None)

    assert configured.INFERENCE.base_url == "https://gateway.example/v1"
    assert configured.INFERENCE.api_key == "inference-key"
    assert configured.chainlit_discovery_endpoint.base_url == "https://lgos.example/v1"
    assert configured.chainlit_discovery_endpoint.api_key == "discovery-key"
    assert configured.chainlit_inference_model("simple") == "lgos/simple"


def test_discovery_requires_a_complete_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_DISCOVERY__BASE_URL", "https://lgos.example/v1")
    monkeypatch.delenv("DEMO_CHAINLIT_DISCOVERY__API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert exc_info.value.errors(include_url=False)[0]["loc"] == (
        "DISCOVERY",
        "api_key",
    )


def test_discovery_defaults_to_inference_client() -> None:
    configured = Settings(
        INFERENCE={
            "base_url": "https://gateway.example/v1",
            "api_key": "shared-key",
        },
    )

    assert configured.DISCOVERY is None
    assert configured.chainlit_discovery_endpoint is configured.INFERENCE
