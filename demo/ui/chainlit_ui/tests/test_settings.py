"""Environment settings coverage for the standalone Chainlit application."""

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from lgos_chainlit.settings import ChainlitSettings, Settings


@pytest.mark.parametrize(
    ("setting", "value", "field"),
    [
        ("OPENAI_GATEWAY_TYPE", "unsupported", "OPENAI_GATEWAY_TYPE"),
        ("OPENAI_GATEWAY_BASE_URL", "ftp://gateway.example", "OPENAI_GATEWAY_BASE_URL"),
        ("DEMO_CHAINLIT_OAUTH_RESOURCE", "relative-resource", "OAUTH_RESOURCE"),
        (
            "DEMO_CHAINLIT_OAUTH_RESOURCE",
            "https://llm.example/#fragment",
            "OAUTH_RESOURCE",
        ),
        ("DEMO_CHAINLIT_OAUTH_ISSUER", "http://id.example", "OAUTH_ISSUER"),
        (
            "DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING",
            "sometimes",
            "ENABLE_OAUTH_TOKEN_FORWARDING",
        ),
        ("DEMO_CHAINLIT_OAUTH_CLIENT_AUTH_METHOD", "none", "OAUTH_CLIENT_AUTH_METHOD"),
        (
            "DEMO_CHAINLIT_OAUTH_ISSUER",
            "https://id.example?issuer=other",
            "OAUTH_ISSUER",
        ),
        (
            "DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS",
            '["invalid-secret-key"]',
            "OAUTH_ENCRYPTION_KEYS",
        ),
    ],
)
def test_settings_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
    field: str,
) -> None:
    monkeypatch.setenv(setting, value)

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert error.value.errors()[0]["loc"] == (field,)
    assert "invalid-secret-key" not in str(error.value)


def test_gateway_settings_read_environment_and_normalize_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "bifrost")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "https://gateway.example/root/")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "api-key")

    configured = Settings(_env_file=None)

    assert configured.OPENAI_GATEWAY_TYPE == "bifrost"
    assert configured.OPENAI_GATEWAY_BASE_URL == "https://gateway.example/root"
    assert configured.OPENAI_GATEWAY_API_KEY == "api-key"


def test_oauth_login_can_use_a_static_gateway_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "shared-key")
    monkeypatch.setenv("DEMO_CHAINLIT_LOGIN_TYPE", "oauth")
    monkeypatch.setenv("DEMO_CHAINLIT_OAUTH_RESOURCE", "https://llm.example/api/")
    monkeypatch.setenv("DEMO_CHAINLIT_OAUTH_ISSUER", "https://id.example")

    configured = Settings(_env_file=None)

    assert configured.OPENAI_GATEWAY_API_KEY == "shared-key"
    assert configured.ENABLE_OAUTH_TOKEN_FORWARDING is False
    assert configured.OAUTH_ENCRYPTION_KEYS == []
    assert configured.OAUTH_RESOURCE == "https://llm.example/api/"

    monkeypatch.delenv("DEMO_CHAINLIT_OAUTH_RESOURCE")
    assert Settings(_env_file=None).OAUTH_RESOURCE is None


@pytest.mark.parametrize("api_key", [None, ""])
def test_oauth_token_forwarding_needs_no_static_gateway_key(
    monkeypatch: pytest.MonkeyPatch, api_key: str | None
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_LOGIN_TYPE", "oauth")
    monkeypatch.setenv("DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING", "true")
    monkeypatch.setenv("DEMO_CHAINLIT_OAUTH_ISSUER", "https://id.example")
    monkeypatch.setenv(
        "DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS",
        '["' + Fernet.generate_key().decode() + '"]',
    )
    if api_key is None:
        monkeypatch.delenv("OPENAI_GATEWAY_API_KEY")
    else:
        monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", api_key)

    configured = Settings(_env_file=None)

    assert configured.OPENAI_GATEWAY_API_KEY is None
    assert configured.ENABLE_OAUTH_TOKEN_FORWARDING is True


def test_oauth_token_forwarding_can_share_the_stack_gateway_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_LOGIN_TYPE", "oauth")
    monkeypatch.setenv("DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING", "true")
    monkeypatch.setenv("DEMO_CHAINLIT_OAUTH_ISSUER", "https://id.example")
    monkeypatch.setenv(
        "DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS",
        '["' + Fernet.generate_key().decode() + '"]',
    )
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "shared-key")

    configured = Settings(_env_file=None)

    assert configured.OPENAI_GATEWAY_API_KEY == "shared-key"
    assert configured.ENABLE_OAUTH_TOKEN_FORWARDING is True


def test_oauth_token_forwarding_requires_oauth_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_LOGIN_TYPE", "mock")
    monkeypatch.setenv("DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING", "true")
    monkeypatch.delenv("OPENAI_GATEWAY_API_KEY")

    with pytest.raises(ValidationError, match="requires OAuth login"):
        Settings(_env_file=None)


def test_oauth_token_forwarding_requires_encryption_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_LOGIN_TYPE", "oauth")
    monkeypatch.setenv("DEMO_CHAINLIT_ENABLE_OAUTH_TOKEN_FORWARDING", "true")
    monkeypatch.setenv("DEMO_CHAINLIT_OAUTH_ISSUER", "https://id.example")
    monkeypatch.delenv("OPENAI_GATEWAY_API_KEY")
    monkeypatch.delenv("DEMO_CHAINLIT_OAUTH_ENCRYPTION_KEYS", raising=False)

    with pytest.raises(ValidationError, match="ENCRYPTION_KEYS must be configured"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "setting",
    [
        "OPENAI_GATEWAY_TYPE",
        "OPENAI_GATEWAY_BASE_URL",
        "OPENAI_GATEWAY_API_KEY",
    ],
)
@pytest.mark.parametrize("value", [None, ""], ids=["missing", "empty"])
def test_gateway_settings_require_nonempty_environment(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(setting)
    else:
        monkeypatch.setenv(setting, value)

    with pytest.raises(ValidationError, match=setting):
        Settings(_env_file=None)


def test_native_chainlit_settings_read_s3_element_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db.example/app")
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "a-secure-test-signing-secret")
    monkeypatch.setenv("BUCKET_NAME", "plots")
    monkeypatch.setenv("APP_AWS_ACCESS_KEY", "access-key")
    monkeypatch.setenv("APP_AWS_SECRET_KEY", "secret-key")
    monkeypatch.setenv("APP_AWS_REGION", "eu-west-1")
    monkeypatch.setenv("DEV_AWS_ENDPOINT", "https://s3.example.com")

    configured = ChainlitSettings(_env_file=None)

    assert configured.BUCKET_NAME == "plots"
    assert configured.DEV_AWS_ENDPOINT == "https://s3.example.com"


def test_configuration_diagnostics_do_not_expose_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "CHAINLIT_AUTH_SECRET": "private-browser-signing-secret",
        "OAUTH_GENERIC_CLIENT_SECRET": "private-oidc-client-secret",
        "APP_AWS_ACCESS_KEY": "private-access-key",
        "APP_AWS_SECRET_KEY": "private-storage-secret",
        "OPENAI_GATEWAY_API_KEY": "private-gateway-key",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:private-password@db.example/app"
    )
    diagnostic = repr(ChainlitSettings(_env_file=None)) + repr(Settings(_env_file=None))
    assert all(value not in diagnostic for value in secrets.values())
    assert "private-password" not in diagnostic

    monkeypatch.setenv("DATABASE_URL", "https://user:private-password@db.example/app")
    with pytest.raises(ValidationError) as error:
        ChainlitSettings(_env_file=None)
    assert "private-password" not in str(error.value)
