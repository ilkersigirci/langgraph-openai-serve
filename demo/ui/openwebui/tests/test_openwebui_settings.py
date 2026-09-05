from lgos_openwebui.functions.generic.gateway import gateway_config
from lgos_openwebui.settings import Settings


def test_settings_fields_have_descriptions() -> None:
    assert all(field.description for field in Settings.model_fields.values())


def test_model_discovery_defaults_to_litellm_hybrid_routing() -> None:
    settings = Settings(_env_file=None)

    assert settings.OPENAI_GATEWAY_TYPE == "litellm"
    assert settings.OPENAI_GATEWAY_BASE_URL is None
    assert settings.API_KEY == "sk-lgos-litellm-demo"
    gateway = gateway_config(settings.OPENAI_GATEWAY_TYPE, local=True)
    assert gateway.responses_base_url == "http://localhost:3007/v1"
    assert gateway.catalog_detail_base_url == "http://localhost:3007/v1"
    assert gateway.files_base_url == "http://localhost:3007/v1"
    assert gateway.files_provider == "litellm_proxy"


def test_bifrost_uses_native_responses_and_catalog_only_passthrough() -> None:
    gateway = gateway_config("bifrost", "https://gateway.example", local=True)

    assert gateway.responses_base_url == "https://gateway.example/openai/v1"
    assert gateway.catalog_base_url == "https://gateway.example/v1"
    assert gateway.catalog_detail_base_url == (
        "https://gateway.example/openai_passthrough/v1"
    )
    assert gateway.files_base_url == "https://gateway.example/v1"
    assert gateway.files_provider == "lgos-files"
