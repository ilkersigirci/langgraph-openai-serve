import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import httpx
import pytest
from langgraph_openai_serve.api.models.schemas import ModelDetails, ModelList

from lgos_demo_api.sync_litellm import sync_models


@pytest.fixture
def model() -> ModelDetails:
    return ModelDetails.model_validate(
        {
            "id": "graph",
            "created": 0,
            "owned_by": "langgraph-openai-serve",
            "lgos": {
                "schema_version": 1,
                "description": "An LGOS graph",
                "features": ["interrupts"],
                "client_settings": {
                    "schema_version": 1,
                    "json_schema": {"type": "object"},
                    "defaults": {"style": "brief"},
                },
            },
        }
    )


@pytest.fixture
def source(model: ModelDetails) -> Iterator[httpx.Client]:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer source-key"
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json=ModelList(data=[model]).model_dump(mode="json")
            )
        assert request.url.path == "/v1/models/graph"
        return httpx.Response(200, json=model.model_dump(mode="json"))

    with httpx.Client(
        base_url="https://source.invalid/v1/",
        headers={"Authorization": "Bearer source-key"},
        transport=httpx.MockTransport(respond),
    ) as client:
        yield client


def test_sync_preserves_operator_settings_and_skips_unchanged_metadata(
    source: httpx.Client, model: ModelDetails
) -> None:
    deployments: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer admin-key"
        if request.method == "GET":
            assert request.url.path == "/model/info"
            return httpx.Response(200, json={"data": deployments})
        payload = json.loads(request.content)
        writes.append(payload)
        if request.method == "POST":
            assert request.url.path == "/model/new"
            deployments.append(deepcopy(payload))
        else:
            assert request.method == "PATCH"
            deployment = deployments[0]
            assert request.url.path == f"/model/{deployment['model_info']['id']}/update"
            # Never round-trip masked credentials or reset operator pricing.
            assert set(payload) == {"model_info"}
            assert set(payload["model_info"]) == {
                "id",
                "db_model",
                "lgos",
                "supports_native_streaming",
            }
            assert payload["model_info"]["id"] == deployment["model_info"]["id"]
            assert payload["model_info"]["db_model"] is True
            deployment["model_info"].update(payload["model_info"])
        return httpx.Response(200, json={})

    with httpx.Client(
        base_url="https://gateway.invalid/",
        headers={"Authorization": "Bearer admin-key"},
        transport=httpx.MockTransport(respond),
    ) as gateway:
        args = {
            "prefix": "research",
            "api_base": "https://graphs.internal/v1",
            "api_key": "source-key",
        }
        assert sync_models(source, gateway, **args) == {"research/graph": "created"}
        deployment = deployments[0]
        assert deployment["model_info"]["lgos"] == model.lgos.model_dump(mode="json")
        assert deployment["litellm_params"] == {
            "model": "openai/graph",
            "api_base": "https://graphs.internal/v1",
            "api_key": "source-key",
            "allowed_openai_params": ["user"],
        }
        deployment["model_info"]["input_cost_per_token"] = 0.00001
        deployment["litellm_params"]["rpm"] = 10
        operator_params = deepcopy(deployment["litellm_params"])
        model.lgos.description = "Updated graph"
        model.lgos.client_settings = None

        assert sync_models(source, gateway, **args) == {"research/graph": "updated"}
        assert deployment["model_info"]["lgos"] == model.lgos.model_dump(mode="json")
        assert deployment["model_info"]["input_cost_per_token"] == 0.00001
        assert deployment["litellm_params"] == operator_params
        assert sync_models(source, gateway, **args) == {"research/graph": "unchanged"}
        assert len(writes) == 2


def test_dry_run_and_failed_discovery_never_write(
    source: httpx.Client, model: ModelDetails
) -> None:
    def read_only_gateway(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("GET", "/model/info")
        return httpx.Response(200, json={"data": []})

    def unavailable_detail(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json=ModelList(data=[model]).model_dump(mode="json")
            )
        return httpx.Response(503)

    with (
        httpx.Client(
            base_url="https://gateway.invalid/",
            transport=httpx.MockTransport(read_only_gateway),
        ) as gateway,
        httpx.Client(
            base_url="https://source.invalid/v1/",
            transport=httpx.MockTransport(unavailable_detail),
        ) as unavailable,
    ):
        args = {
            "prefix": "research",
            "api_base": "https://graphs.internal/v1",
            "api_key": "source-key",
        }
        assert sync_models(source, gateway, dry_run=True, **args) == {
            "research/graph": "created"
        }
        with pytest.raises(httpx.HTTPStatusError):
            sync_models(unavailable, gateway, **args)


@pytest.mark.parametrize("db_model,count", [(False, 1), (True, 2)])
def test_ambiguous_or_config_owned_deployments_are_not_modified(
    source: httpx.Client, db_model: bool, count: int
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "model_name": "research/graph",
                        "model_info": {"id": "existing-id", "db_model": db_model},
                    }
                ]
                * count
            },
        )

    with (
        httpx.Client(
            base_url="https://gateway.invalid/", transport=httpx.MockTransport(respond)
        ) as gateway,
        pytest.raises(ValueError, match="database-backed deployment"),
    ):
        sync_models(
            source,
            gateway,
            prefix="research",
            api_base="https://graphs.internal/v1",
            api_key="source-key",
        )


@pytest.mark.parametrize("prefix", ["", " ", "research team", "a/b", "a*"])
def test_invalid_namespace_is_rejected_before_discovery(prefix: str) -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Invalid namespace must not make requests: {request.url}")

    with (
        httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client,
        pytest.raises(ValueError, match="Model namespace"),
    ):
        sync_models(client, client, prefix=prefix, api_base="unused", api_key="unused")
