import pytest
from demo.api.settings import Settings


def test_demo_settings_use_demo_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "POSTGRES_URI": "postgresql://demo:test@db/demo",
        "OPENAI_BASE_URL": "https://example.com/v1",
        "OPENAI_API_KEY": "demo-key",
        "OPENAI_MODEL": "demo-model",
        "CHAINLIT_OPENAI_BASE_URL": "https://chainlit.example.com/v1",
        "CHAINLIT_HITL_MODEL": "approval-model",
    }
    for name, value in expected.items():
        monkeypatch.setenv(f"DEMO_{name}", value)

    settings = Settings(_env_file=None)

    assert settings.model_dump() == expected
