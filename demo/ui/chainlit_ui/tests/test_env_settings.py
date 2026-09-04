"""Environment settings coverage for the standalone Chainlit application."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from openai import OpenAIError
from openai.types import Model
from pydantic import ValidationError

from lgos_chainlit.settings import ChainlitSettings, Settings
from lgos_chainlit.utils import clients


def test_ui_file_rejects_unknown_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_UI_FILE", "../other")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_endpoint_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_OPENAI__BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv(
        "DEMO_CHAINLIT_OPENAI__CATALOG_BASE_URL",
        "https://gateway.example/catalog/v1",
    )
    monkeypatch.setenv(
        "DEMO_CHAINLIT_OPENAI__FILES_BASE_URL",
        "https://files.example/v1",
    )
    monkeypatch.setenv("DEMO_CHAINLIT_OPENAI__API_KEY", "api-key")
    monkeypatch.setenv("DEMO_CHAINLIT_OPENAI__FILES_PROVIDER", "lgos-files")

    configured = Settings(_env_file=None)

    assert configured.OPENAI.base_url == "https://gateway.example/v1"
    assert configured.OPENAI.catalog_base_url == "https://gateway.example/catalog/v1"
    assert configured.OPENAI.files_base_url == "https://files.example/v1"
    assert configured.OPENAI.files_provider == "lgos-files"
    assert configured.OPENAI.api_key == "api-key"


def test_files_endpoint_defaults_to_the_standalone_service() -> None:
    configured = Settings(_env_file=None)

    assert configured.OPENAI.files_base_url == "http://localhost:3006/v1"


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
        clients.settings.OPENAI,
        "catalog_base_url",
        "https://gateway.example/v1",
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

    passthrough_list = AsyncMock(side_effect=list_provider_models)
    monkeypatch.setattr(clients.catalog_client.models, "list", catalog_list)
    monkeypatch.setattr(clients.openai_client.models, "list", passthrough_list)

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
    passthrough_list.assert_has_awaits(
        [
            call(extra_headers={"x-model-provider": "lgos-a"}),
            call(extra_headers={"x-model-provider": "lgos-future"}),
        ],
        any_order=True,
    )
    assert passthrough_list.await_count == 2


async def test_standard_endpoint_preserves_listed_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clients.settings.OPENAI, "catalog_base_url", None)
    list_models = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                Model(
                    id="lgos-b/namespace/graph-b",
                    object="model",
                    created=1,
                    owned_by="test",
                )
            ]
        )
    )
    monkeypatch.setattr(clients.openai_client.models, "list", list_models)

    assert [model.id for model in await clients.list_models()] == [
        "lgos-b/namespace/graph-b"
    ]
    list_models.assert_awaited_once_with()
    assert clients.model_request("lgos-b/namespace/graph-b") == {
        "model": "lgos-b/namespace/graph-b"
    }


def test_bifrost_model_request_requires_a_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients.settings.OPENAI,
        "catalog_base_url",
        "https://gateway.example/v1",
    )

    with pytest.raises(ValueError, match="provider/model"):
        clients.model_request("graph-b")


def test_chat_client_identifies_chainlit_for_telemetry() -> None:
    assert clients.openai_client.default_headers["User-Agent"] == "lgos-chainlit"


async def test_model_retrieval_rejects_a_non_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients.openai_client.models,
        "retrieve",
        AsyncMock(return_value="unsupported model detail"),
    )

    with pytest.raises(OpenAIError, match="invalid model"):
        await clients.retrieve_model("simple-graph")
