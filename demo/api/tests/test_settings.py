import pytest

from lgos_demo_api.settings import Settings


def test_settings_read_demo_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_uri = "postgresql://demo:test@db/demo"
    files_base_url = "https://files.example.com/v1"
    monkeypatch.setenv("DEMO_API_POSTGRES_URI", postgres_uri)
    monkeypatch.setenv("DEMO_API_FILES_BASE_URL", files_base_url)

    settings = Settings(_env_file=None)

    assert postgres_uri == settings.POSTGRES_URI
    assert files_base_url == settings.FILES_BASE_URL
