from pathlib import Path

import pytest

from {{ cookiecutter.project_slug }}.settings import (
    Settings,
)


def test_process_environment_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "APP_OPENAI_BASE_URL=https://gateway.example.com/v1\n"
        "APP_OPENAI_MODEL=file-model\n"
    )
    monkeypatch.setenv("APP_OPENAI_MODEL", "environment-model")
    monkeypatch.delenv("APP_OPENAI_BASE_URL", raising=False)
    settings = Settings(_env_file=dotenv)
    assert settings.OPENAI_BASE_URL == "https://gateway.example.com/v1"
    assert settings.OPENAI_MODEL == "environment-model"
