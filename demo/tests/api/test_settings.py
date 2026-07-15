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
