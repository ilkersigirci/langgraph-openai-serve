"""Environment settings coverage for the standalone Chainlit application."""

import httpx
import pytest
from cryptography.fernet import Fernet
from openai import OpenAIError
from pydantic import ValidationError

from lgos_chainlit.gateway import gateway_config
from lgos_chainlit.settings import ChainlitSettings, Settings
from lgos_chainlit.utils import clients


@pytest.mark.parametrize(
    ("setting", "value", "field"),
    [
        ("DEMO_CHAINLIT_UI_FILE", "../other", "UI_FILE"),
        ("OPENAI_GATEWAY_TYPE", "unsupported", "OPENAI_GATEWAY_TYPE"),
        ("OPENAI_GATEWAY_BASE_URL", "ftp://gateway.example", "OPENAI_GATEWAY_BASE_URL"),
        ("DEMO_CHAINLIT_OAUTH_RESOURCE", "relative-resource", "OAUTH_RESOURCE"),
        (
            "DEMO_CHAINLIT_OAUTH_RESOURCE",
            "https://llm.example/#fragment",
            "OAUTH_RESOURCE",
        ),
        ("DEMO_CHAINLIT_OAUTH_ISSUER", "http://id.example", "OAUTH_ISSUER"),
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


@pytest.mark.parametrize("api_key", [None, ""])
def test_oauth_gateway_resource_is_optional_and_needs_no_shared_key(
    monkeypatch: pytest.MonkeyPatch, api_key: str | None
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_LOGIN_TYPE", "oauth")
    monkeypatch.setenv("DEMO_CHAINLIT_OAUTH_RESOURCE", "https://llm.example/api/")
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
    assert configured.OAUTH_RESOURCE == "https://llm.example/api/"

    monkeypatch.delenv("DEMO_CHAINLIT_OAUTH_RESOURCE")
    assert Settings(_env_file=None).OAUTH_RESOURCE is None


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


async def test_bifrost_catalog_preserves_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients, "gateway", gateway_config("bifrost", "https://gateway.example")
    )
    graph = {
        "id": "graph",
        "object": "model",
        "created": 1,
        "owned_by": "langgraph-openai-serve",
        "lgos": {"schema_version": 1, "description": "Graph", "features": []},
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            data = [
                {**graph, "id": "team/graph"},
                {**graph, "id": "other/graph"},
                {**graph, "id": "gpt-5", "owned_by": "openai"},
            ]
        else:
            assert request.url.path == "/openai_passthrough/v1/models"
            assert request.headers["x-model-provider"] in {"team", "other"}
            data = [graph]
        return httpx.Response(200, json={"object": "list", "data": data})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        models = await clients.list_models()

    assert [model.id for model in models] == ["other/graph", "team/graph"]
    assert (models[0].model_extra or {})["lgos"] == graph["lgos"]


def test_bifrost_uses_native_responses_and_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = gateway_config("bifrost", "https://gateway.example")
    monkeypatch.setattr(clients, "gateway", gateway)

    assert gateway.responses_base_url == "https://gateway.example/openai/v1"
    assert gateway.files_base_url == "https://gateway.example/v1"
    assert gateway.files_provider == "lgos-files"
    assert clients.model_request("lgos-b/namespace/graph-b") == {
        "model": "namespace/graph-b",
        "extra_headers": {"x-model-provider": "lgos-b"},
    }
    with pytest.raises(ValueError, match="provider/model"):
        clients.model_request("graph-b")


def test_chat_client_identifies_chainlit_for_telemetry() -> None:
    assert clients.openai_client.default_headers["User-Agent"] == "lgos-chainlit"


async def test_model_retrieval_rejects_a_non_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients, "gateway", gateway_config("bifrost", "https://gateway.example")
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json="unsupported model detail")
        )
    ) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        with pytest.raises(OpenAIError, match="invalid model"):
            await clients.retrieve_model("lgos-a/simple-graph")


async def test_litellm_model_info_owns_catalog_and_preserves_public_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients, "gateway", gateway_config("litellm", "https://gateway.example")
    )
    metadata = {"schema_version": 1, "description": "Graph", "features": []}
    names = ["graph", "research/namespace/graph"]
    deployments = [
        {"model_name": name, "model_info": {"lgos": metadata}} for name in names
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/model/info"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert "x-model-provider" not in request.headers
        return httpx.Response(
            200,
            json={
                "data": [
                    *deployments,
                    deployments[0],
                    {"model_name": "gpt-5", "model_info": {}},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http, api_key="test-key"),
        )
        models = await clients.list_models()
        retrieved = await clients.retrieve_model(names[1])
        with pytest.raises(OpenAIError, match="not available"):
            await clients.retrieve_model("removed")

    assert [model.id for model in models] == names
    assert retrieved.id == names[1]
    assert (retrieved.model_extra or {})["lgos"] == metadata
    for name in names:
        assert clients.model_request(name) == {"model": name}


@pytest.mark.parametrize("status", [403, 200], ids=["forbidden", "invalid-payload"])
async def test_litellm_catalog_errors_do_not_fall_back_to_other_routes(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    monkeypatch.setattr(
        clients, "gateway", gateway_config("litellm", "https://gateway.example")
    )

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/model/info"
        return httpx.Response(status, json={"error": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        with pytest.raises(OpenAIError if status == 403 else ValidationError):
            await clients.list_models()
