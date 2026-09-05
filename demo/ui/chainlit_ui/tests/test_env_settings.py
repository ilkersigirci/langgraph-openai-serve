"""Environment settings coverage for the standalone Chainlit application."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from openai import OpenAIError
from openai.types import Model
from pydantic import ValidationError

from lgos_chainlit.gateway import gateway_config
from lgos_chainlit.settings import ChainlitSettings, Settings
from lgos_chainlit.utils import clients


def test_ui_file_rejects_unknown_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_UI_FILE", "../other")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_endpoint_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "bifrost")
    monkeypatch.setenv(
        "DEMO_CHAINLIT_OPENAI__GATEWAY_BASE_URL",
        "https://gateway.example",
    )
    monkeypatch.setenv("DEMO_CHAINLIT_OPENAI__API_KEY", "api-key")

    configured = Settings(_env_file=None)

    assert configured.OPENAI_GATEWAY_TYPE == "bifrost"
    assert configured.OPENAI.gateway_base_url == "https://gateway.example"
    assert configured.OPENAI.api_key == "api-key"


def test_openai_endpoints_default_to_litellm_managed_responses() -> None:
    configured = Settings(_env_file=None)

    assert configured.OPENAI_GATEWAY_TYPE == "litellm"
    assert configured.OPENAI.gateway_base_url is None
    assert configured.OPENAI.api_key == "sk-lgos-litellm-demo"
    gateway = gateway_config(configured.OPENAI_GATEWAY_TYPE)
    assert gateway.responses_base_url == "http://localhost:3007/v1"
    assert gateway.catalog_detail_base_url == "http://localhost:3007/v1"
    assert gateway.files_base_url == "http://localhost:3007/v1"
    assert gateway.files_provider == "litellm_proxy"


def test_native_chainlit_settings_require_s3_element_storage(
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


async def test_catalog_discovers_providers_and_preserves_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients,
        "gateway",
        gateway_config("bifrost", "https://gateway.example"),
    )
    catalog_list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                Model(
                    id="lgos-a/graph-a",
                    object="model",
                    created=1,
                    owned_by="langgraph-openai-serve",
                ),
                Model(
                    id="lgos-future/graph-b",
                    object="model",
                    created=1,
                    owned_by="langgraph-openai-serve",
                ),
                Model(
                    id="gpt-5",
                    object="model",
                    created=1,
                    owned_by="openai",
                ),
            ]
        )
    )
    provider_models = {
        "lgos-a": SimpleNamespace(
            data=[
                Model(
                    id="graph-a",
                    object="model",
                    created=1,
                    owned_by="langgraph-openai-serve",
                    langgraph_openai_serve={
                        "schema_version": 1,
                        "description": "DUMMY",
                        "features": [],
                    },
                )
            ]
        ),
        "lgos-future": SimpleNamespace(
            data=[
                Model(
                    id="graph-b",
                    object="model",
                    created=1,
                    owned_by="langgraph-openai-serve",
                )
            ]
        ),
    }

    def list_provider_models(*, extra_headers: dict[str, str]) -> SimpleNamespace:
        return provider_models[extra_headers["x-model-provider"]]

    api_list = AsyncMock(side_effect=list_provider_models)
    monkeypatch.setattr(clients.catalog_client.models, "list", catalog_list)
    monkeypatch.setattr(clients.catalog_detail_client.models, "list", api_list)

    models = await clients.list_models()

    assert [model.id for model in models] == [
        "lgos-a/graph-a",
        "lgos-future/graph-b",
    ]
    assert (models[0].model_extra or {})["langgraph_openai_serve"] == {
        "schema_version": 1,
        "description": "DUMMY",
        "features": [],
    }
    catalog_list.assert_awaited_once_with()
    api_list.assert_has_awaits(
        [
            call(extra_headers={"x-model-provider": "lgos-a"}),
            call(extra_headers={"x-model-provider": "lgos-future"}),
        ],
        any_order=True,
    )
    assert api_list.await_count == 2


def test_bifrost_uses_native_responses_and_catalog_only_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = gateway_config("bifrost", "https://gateway.example")
    monkeypatch.setattr(clients, "gateway", gateway)

    assert gateway.responses_base_url == "https://gateway.example/openai/v1"
    assert gateway.catalog_base_url == "https://gateway.example/v1"
    assert gateway.catalog_detail_base_url == (
        "https://gateway.example/openai_passthrough/v1"
    )
    assert gateway.files_base_url == "https://gateway.example/v1"
    assert gateway.files_provider == "lgos-files"
    assert clients.model_request("lgos-b/namespace/graph-b") == {
        "model": "namespace/graph-b",
        "extra_headers": {"x-model-provider": "lgos-b"},
    }


def test_bifrost_model_request_requires_a_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients,
        "gateway",
        gateway_config("bifrost", "https://gateway.example"),
    )

    with pytest.raises(ValueError, match="provider/model"):
        clients.model_request("graph-b")


def test_chat_client_identifies_chainlit_for_telemetry() -> None:
    assert clients.openai_client.default_headers["User-Agent"] == "lgos-chainlit"


async def test_model_retrieval_rejects_a_non_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients,
        "gateway",
        gateway_config("bifrost", "https://gateway.example"),
    )
    monkeypatch.setattr(
        clients.catalog_detail_client.models,
        "retrieve",
        AsyncMock(return_value="unsupported model detail"),
    )

    with pytest.raises(OpenAIError, match="invalid model"):
        await clients.retrieve_model("lgos-a/simple-graph")


async def test_litellm_catalog_prefixes_models_and_owns_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients,
        "gateway",
        gateway_config("litellm", "https://gateway.example"),
    )
    catalog_clients = {}
    for model_prefix in ("lgos-a", "lgos-b"):
        model = Model(
            id="graph",
            object="model",
            created=1,
            owned_by="langgraph-openai-serve",
            langgraph_openai_serve={
                "schema_version": 1,
                "description": f"Graph {model_prefix}",
                "features": [],
            },
        )
        catalog_clients[model_prefix] = SimpleNamespace(
            models=SimpleNamespace(
                list=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[
                            model,
                            Model(
                                id="gpt-5",
                                object="model",
                                created=1,
                                owned_by="openai",
                            ),
                        ]
                    )
                ),
                retrieve=AsyncMock(return_value=model),
            )
        )
    monkeypatch.setattr(
        clients,
        "_catalog_client",
        lambda model_prefix: catalog_clients[model_prefix],
    )

    models = await clients.list_models()
    retrieved = await clients.retrieve_model("lgos-b/graph")

    assert [listed.id for listed in models] == ["lgos-a/graph", "lgos-b/graph"]
    assert retrieved.id == "graph"
    assert clients.model_request("graph") == {"model": "lgos-a/graph"}
    assert clients.model_request("lgos-b/graph") == {"model": "lgos-b/graph"}
    for catalog in catalog_clients.values():
        catalog.models.list.assert_awaited_once_with()
    catalog_clients["lgos-b"].models.retrieve.assert_awaited_once_with(model="graph")


def test_litellm_model_request_rejects_an_unknown_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients,
        "gateway",
        gateway_config("litellm", "https://gateway.example"),
    )

    with pytest.raises(ValueError, match="lgos-a/model, lgos-b/model"):
        clients.model_request("lgos-c/graph-c")
