import pytest

from lgos_demo_api.core.settings import Settings


def test_settings_read_demo_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files_base_url = "https://files.example.com/v1"
    monkeypatch.setenv("DEMO_API_FILES_BASE_URL", files_base_url)
    monkeypatch.setenv("DEMO_API_WEB_SEARCH_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "https://gateway.example/demo/")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "gateway-key")

    settings = Settings(_env_file=None)

    assert files_base_url == settings.FILES_BASE_URL
    assert settings.WEB_SEARCH_BACKEND == "openai"
    assert settings.openai_base_url == "https://gateway.example/demo/v1"
    assert settings.vector_store_base_url == (
        "https://gateway.example/demo/openai_passthrough/v1"
    )
    assert settings.OPENAI_GATEWAY_API_KEY == "gateway-key"
