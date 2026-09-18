"""Gateway routing and OpenAI client coverage."""

import httpx2
import pytest
from openai import OpenAIError
from pydantic import ValidationError

from lgos_chainlit import clients
from lgos_chainlit.gateway import gateway_config


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

    def handle(request: httpx2.Request) -> httpx2.Response:
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
        return httpx2.Response(200, json={"object": "list", "data": data})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
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
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(200, json="unsupported model detail")
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

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        assert request.url.path == "/model/info"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert "x-model-provider" not in request.headers
        return httpx2.Response(
            200,
            json={
                "data": [
                    *deployments,
                    deployments[0],
                    {"model_name": "gpt-5", "model_info": {}},
                ]
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
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

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/model/info"
        return httpx2.Response(status, json={"error": "unavailable"})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
        monkeypatch.setattr(
            clients,
            "openai_client",
            clients.openai_client.with_options(http_client=http),
        )
        with pytest.raises(OpenAIError if status == 403 else ValidationError):
            await clients.list_models()
