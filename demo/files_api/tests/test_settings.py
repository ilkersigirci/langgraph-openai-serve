import pytest

from lgos_files_api.settings import Settings


def test_settings_read_files_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_API_FILES_AWS_ACCESS_KEY_ID", "files-access")
    monkeypatch.setenv("DEMO_API_FILES_AWS_SECRET_ACCESS_KEY", "files-secret")
    monkeypatch.setenv("DEMO_API_FILES_AWS_DEFAULT_REGION", "files-region")

    settings = Settings(_env_file=None)

    assert settings.AWS_ACCESS_KEY_ID == "files-access"
    assert settings.AWS_SECRET_ACCESS_KEY == "files-secret"
    assert settings.AWS_DEFAULT_REGION == "files-region"
