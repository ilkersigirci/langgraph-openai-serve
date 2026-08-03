"""Environment settings coverage for the standalone Chainlit application."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from openai import OpenAIError
from openai.types import Model
from pydantic import ValidationError

from lgos_chainlit.settings import Settings
from lgos_chainlit.utils import clients


def test_ui_file_rejects_unknown_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_UI_FILE", "../other")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_endpoint_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_CHAINLIT_OPENAI__BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("DEMO_CHAINLIT_OPENAI__API_KEY", "api-key")
    monkeypatch.setenv(
        "DEMO_CHAINLIT_OPENAI__MODEL_ROUTES",
        '{"lgos-a":{"x-route":"a"},"lgos-b":{"x-route":"b"}}',
    )

    configured = Settings(_env_file=None)

    assert configured.OPENAI.base_url == "https://gateway.example/v1"
    assert configured.OPENAI.api_key == "api-key"
    assert configured.OPENAI.model_routes == {
        "lgos-a": {"x-route": "a"},
        "lgos-b": {"x-route": "b"},
    }


@pytest.mark.anyio
async def test_one_client_lists_two_explicit_model_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clients.settings.OPENAI,
        "model_routes",
        {
            "lgos-a": {"x-route": "a"},
            "lgos-b": {"x-route": "b"},
        },
    )
    list_models = AsyncMock(
        side_effect=[
            SimpleNamespace(
                data=[
                    Model(
                        id="graph-a",
                        object="model",
                        created=1,
                        owned_by="test",
                        langgraph_openai_serve={
                            "schema_version": 1,
                            "description": "DUMMY",
                        },
                    )
                ]
            ),
            SimpleNamespace(
                data=[
                    Model(
                        id="graph-b",
                        object="model",
                        created=1,
                        owned_by="test",
                    )
                ]
            ),
        ]
    )
    monkeypatch.setattr(clients.openai_client.models, "list", list_models)

    models = await clients.list_models()

    assert [model.id for model in models] == ["lgos-a/graph-a", "lgos-b/graph-b"]
    assert (models[0].model_extra or {})["langgraph_openai_serve"] == {
        "schema_version": 1,
        "description": "DUMMY",
    }
    assert list_models.await_args_list == [
        call(extra_headers={"x-route": "a"}),
        call(extra_headers={"x-route": "b"}),
    ]
    assert clients.model_request("lgos-b/namespace/graph-b") == {
        "model": "namespace/graph-b",
        "extra_headers": {"x-route": "b"},
    }
    with pytest.raises(ValueError, match="model route"):
        clients.model_request("graph-b")


@pytest.mark.anyio
async def test_standard_endpoint_preserves_listed_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clients.settings.OPENAI, "model_routes", {})
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


@pytest.mark.anyio
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
