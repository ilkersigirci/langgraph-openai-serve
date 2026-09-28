import pytest
from pydantic import ValidationError

from lgos_openwebui.functions.generic.gateway import gateway_config
from lgos_openwebui.functions.generic.pipe import Pipe
from lgos_openwebui.settings import Settings


@pytest.mark.parametrize(
    "setting",
    [
        "OPENAI_GATEWAY_TYPE",
        "OPENAI_GATEWAY_BASE_URL",
        "OPENAI_GATEWAY_API_KEY",
    ],
)
@pytest.mark.parametrize("value", [None, ""], ids=["missing", "empty"])
def test_sync_settings_require_nonempty_environment(
    gateway_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(setting)
    else:
        monkeypatch.setenv(setting, value)

    with pytest.raises(ValidationError, match=setting):
        Settings()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("OPENAI_GATEWAY_TYPE", "unsupported"),
        ("OPENAI_GATEWAY_BASE_URL", "ftp://gateway.example"),
    ],
)
def test_sync_settings_reject_invalid_values(
    gateway_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
) -> None:
    monkeypatch.setenv(setting, value)

    with pytest.raises(ValidationError) as error:
        Settings()
    assert error.value.errors()[0]["loc"] == (setting,)


def test_sync_settings_read_environment_and_normalize_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "bifrost")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "https://gateway.example/root/")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "api-key")

    settings = Settings()

    assert settings.OPENAI_GATEWAY_TYPE == "bifrost"
    assert settings.OPENAI_GATEWAY_BASE_URL == "https://gateway.example/root"
    assert settings.OPENAI_GATEWAY_API_KEY == "api-key"


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("OPENAI_GATEWAY_TYPE", "unsupported"),
        ("OPENAI_GATEWAY_BASE_URL", "ftp://gateway.example"),
        ("OPENAI_GATEWAY_API_KEY", ""),
    ],
)
def test_gateway_valves_reject_invalid_values(setting: str, value: str) -> None:
    with pytest.raises(ValidationError) as error:
        Pipe.Valves(**{setting: value})
    assert error.value.errors()[0]["loc"][0] == setting


def test_gateway_valves_normalize_root() -> None:
    valves = Pipe.Valves(OPENAI_GATEWAY_BASE_URL="https://gateway.example/root/")

    assert valves.OPENAI_GATEWAY_BASE_URL == "https://gateway.example/root"


def test_gateway_valves_keep_typed_admin_form_inputs() -> None:
    # Open WebUI's valves form reads only these top-level schema keys.
    properties = Pipe.Valves.model_json_schema()["properties"]

    assert properties["OPENAI_GATEWAY_TYPE"]["enum"] == ["litellm", "bifrost"]
    assert properties["OPENAI_GATEWAY_BASE_URL"]["type"] == "string"
    assert properties["OPENAI_GATEWAY_API_KEY"]["type"] == "string"
    assert properties["OPENAI_GATEWAY_API_KEY"]["input"] == {"type": "password"}


def test_bifrost_uses_native_responses_and_files() -> None:
    gateway = gateway_config("bifrost", "https://gateway.example")

    assert gateway.responses_base_url == "https://gateway.example/openai/v1"
    assert gateway.files_base_url == "https://gateway.example/v1"
    assert gateway.files_provider == "lgos-files"
