"""Gateway routing and model catalog coverage."""

import json

import httpx2
import pytest
from openai import OpenAIError
from pydantic import ValidationError

from lgos_chainlit import clients
from lgos_chainlit.gateway import gateway_config


@pytest.mark.parametrize(
    ("gateway_type", "responses_base_url", "files_provider"),
    [
        ("litellm", "https://gateway.example/v1", "litellm_proxy"),
        ("bifrost", "https://gateway.example/openai/v1", "lgos-files"),
    ],
)
def test_each_gateway_uses_its_native_responses_and_files_routing(
    gateway_type: str,
    responses_base_url: str,
    files_provider: str,
) -> None:
    gateway = gateway_config(gateway_type, "https://gateway.example/")

    assert (gateway.responses_base_url, gateway.files_provider) == (
        responses_base_url,
        files_provider,
    )


async def test_bifrost_catalog_preserves_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
    fake_gateway,
) -> None:
    monkeypatch.setattr(
        clients, "gateway", gateway_config("bifrost", "https://gateway.example")
    )
    graph = {
        "id": "graph",
        "object": "model",
        "created": 1,
        "owned_by": "langgraph-openai-serve",
        "additional_attributes": {
            "description": "Graph",
            "lgos": json.dumps({"description": "Graph", "features": []}),
        },
    }
    catalog = [
        {**graph, "id": "lgos/graph"},
        {**graph, "id": "lgos/coding-agent", "owned_by": None},
        {**graph, "id": "gpt-5", "owned_by": "openai"},
    ]
    fake_gateway.replies += [
        httpx2.Response(200, json={"object": "list", "data": catalog})
    ]

    models = await clients.list_models()

    assert [model.id for model in models] == ["lgos/graph", "lgos/coding-agent"]
    assert models[1].owned_by == "langgraph-openai-serve"
    assert (models[1].model_extra or {})["lgos"] == {
        "description": "Graph",
        "features": [],
    }
    assert [request.url.path for request in fake_gateway.requests] == ["/v1/models"]


async def test_model_retrieval_rejects_a_model_missing_from_the_native_catalog(
    monkeypatch: pytest.MonkeyPatch,
    fake_gateway,
) -> None:
    monkeypatch.setattr(
        clients, "gateway", gateway_config("bifrost", "https://gateway.example")
    )
    fake_gateway.replies.append(
        httpx2.Response(200, json={"object": "list", "data": []})
    )

    with pytest.raises(OpenAIError, match="not available"):
        await clients.retrieve_model("lgos/simple-graph")


async def test_litellm_model_info_owns_catalog_and_preserves_public_names(
    fake_gateway,
) -> None:
    metadata = {"description": "Graph", "features": []}
    names = ["graph", "research/namespace/graph"]
    deployments = [
        {"model_name": name, "model_info": {"lgos": metadata}} for name in names
    ]
    payload = {
        "data": [
            *deployments,
            deployments[0],
            {"model_name": "gpt-5", "model_info": {}},
        ]
    }
    fake_gateway.replies += [httpx2.Response(200, json=payload) for _ in range(3)]

    models = await clients.list_models()
    retrieved = await clients.retrieve_model(names[1])
    with pytest.raises(OpenAIError, match="not available"):
        await clients.retrieve_model("removed")

    assert [model.id for model in models] == names
    assert retrieved.id == names[1]
    assert (retrieved.model_extra or {})["lgos"] == metadata
    assert {
        (request.method, request.url.path, request.headers["Authorization"])
        for request in fake_gateway.requests
    } == {("GET", "/model/info", "Bearer test-api-key")}


@pytest.mark.parametrize("status", [403, 200], ids=["forbidden", "invalid-payload"])
async def test_litellm_catalog_errors_do_not_fall_back_to_other_routes(
    fake_gateway,
    status: int,
) -> None:
    fake_gateway.replies.append(httpx2.Response(status, json={"error": "unavailable"}))

    with pytest.raises(OpenAIError if status == 403 else ValidationError):
        await clients.list_models()

    assert [request.url.path for request in fake_gateway.requests] == ["/model/info"]
