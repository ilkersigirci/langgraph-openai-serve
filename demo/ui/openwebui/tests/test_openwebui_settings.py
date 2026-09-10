import pytest
from pydantic import ValidationError

from lgos_openwebui.functions.generic.gateway import gateway_config
from lgos_openwebui.functions.generic.pipe import Pipe
from lgos_openwebui.settings import Settings


@pytest.mark.parametrize(
    ("settings_model", "error_type"),
    [
        pytest.param(Settings, ValidationError, id="sync"),
        pytest.param(Pipe.Valves, RuntimeError, id="valves"),
    ],
)
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
    gateway_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    settings_model: type[Settings] | type[Pipe.Valves],
    error_type: type[Exception],
    setting: str,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(setting)
    else:
        monkeypatch.setenv(setting, value)

    with pytest.raises(error_type, match=setting):
        settings_model()


@pytest.mark.parametrize(
    "settings_model", [Settings, Pipe.Valves], ids=["sync", "valves"]
)
@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("OPENAI_GATEWAY_TYPE", "unsupported"),
        ("OPENAI_GATEWAY_BASE_URL", "ftp://gateway.example"),
    ],
)
def test_gateway_settings_reject_invalid_values(
    gateway_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    settings_model: type[Settings] | type[Pipe.Valves],
    setting: str,
    value: str,
) -> None:
    monkeypatch.setenv(setting, value)

    with pytest.raises(ValidationError) as error:
        settings_model()
    assert error.value.errors()[0]["loc"] == (setting,)


@pytest.mark.parametrize(
    "settings_model", [Settings, Pipe.Valves], ids=["sync", "valves"]
)
def test_gateway_settings_read_environment_and_normalize_root(
    monkeypatch: pytest.MonkeyPatch,
    settings_model: type[Settings] | type[Pipe.Valves],
) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "bifrost")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "https://gateway.example/root/")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "api-key")

    settings = settings_model()

    assert settings.OPENAI_GATEWAY_TYPE == "bifrost"
    assert settings.OPENAI_GATEWAY_BASE_URL == "https://gateway.example/root"
    assert settings.OPENAI_GATEWAY_API_KEY == "api-key"


def test_gateway_url_has_a_string_schema_for_the_valves_form() -> None:
    schema = Pipe.Valves.model_json_schema()

    assert schema["properties"]["OPENAI_GATEWAY_BASE_URL"]["type"] == "string"


def test_bifrost_uses_native_responses_and_catalog_only_passthrough() -> None:
    gateway = gateway_config("bifrost", "https://gateway.example")

    assert gateway.responses_base_url == "https://gateway.example/openai/v1"
    assert gateway.catalog_base_url == "https://gateway.example/v1"
    assert gateway.catalog_detail_base_url == (
        "https://gateway.example/openai_passthrough/v1"
    )
    assert gateway.files_base_url == "https://gateway.example/v1"
    assert gateway.files_provider == "lgos-files"
