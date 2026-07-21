import pytest

from lgos_demo_api.settings import Settings


def test_settings_read_demo_prefixed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_uri = "postgresql://demo:test@db/demo"
    monkeypatch.setenv("DEMO_API_POSTGRES_URI", postgres_uri)

    settings = Settings(_env_file=None)

    assert postgres_uri == settings.POSTGRES_URI
