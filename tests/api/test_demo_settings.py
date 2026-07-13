import pytest
from demo.api.settings import Settings


def test_demo_settings_use_demo_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "postgresql://demo:test@db/demo"
    monkeypatch.setenv("DEMO_POSTGRES_URI", expected)

    settings = Settings(_env_file=None)

    assert settings.POSTGRES_URI == expected
