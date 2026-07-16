import pytest
from demo.api.settings import Settings
from pydantic import ValidationError


def test_settings_read_demo_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_uri = "postgresql://demo:test@db/demo"
    monkeypatch.setenv("DEMO_POSTGRES_URI", postgres_uri)

    settings = Settings()

    assert postgres_uri == settings.POSTGRES_URI


def test_chainlit_ui_file_rejects_unknown_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_UI_FILE", "../other")

    with pytest.raises(ValidationError):
        Settings()


def test_chainlit_proxy_discovery_and_inference_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DEMO_CHAINLIT_INFERENCE__BASE_URL", "https://gateway.example/v1"
    )
    monkeypatch.setenv("DEMO_CHAINLIT_INFERENCE__API_KEY", "inference-key")
    monkeypatch.setenv("DEMO_CHAINLIT_DISCOVERY__BASE_URL", "https://lgos.example/v1")
    monkeypatch.setenv("DEMO_CHAINLIT_DISCOVERY__API_KEY", "discovery-key")
    monkeypatch.setenv("DEMO_CHAINLIT_INFERENCE_MODEL_PREFIX", "lgos/")

    settings = Settings(_env_file=None)

    assert settings.CHAINLIT_INFERENCE.base_url == "https://gateway.example/v1"
    assert settings.CHAINLIT_INFERENCE.api_key == "inference-key"
    assert settings.chainlit_discovery_endpoint.base_url == "https://lgos.example/v1"
    assert settings.chainlit_discovery_endpoint.api_key == "discovery-key"
    assert settings.chainlit_inference_model("simple") == "lgos/simple"


def test_chainlit_discovery_requires_a_complete_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_DISCOVERY__BASE_URL", "https://lgos.example/v1")
    monkeypatch.delenv("DEMO_CHAINLIT_DISCOVERY__API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert exc_info.value.errors(include_url=False)[0]["loc"] == (
        "CHAINLIT_DISCOVERY",
        "api_key",
    )


def test_chainlit_discovery_defaults_to_inference_client() -> None:
    settings = Settings(
        CHAINLIT_INFERENCE={
            "base_url": "https://gateway.example/v1",
            "api_key": "shared-key",
        },
    )

    assert settings.CHAINLIT_DISCOVERY is None
    assert settings.chainlit_discovery_endpoint is settings.CHAINLIT_INFERENCE
