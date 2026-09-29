import json
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from langgraph_openai_serve.api.models.schemas import (
    LangGraphModelExtension,
    ModelDetails,
    ModelList,
)

from lgos_demo_api.utils.sync_bifrost import (
    ModelAttributes,
    prepare_catalog,
    sync_catalog,
)


@pytest.fixture
def model() -> ModelDetails:
    return ModelDetails.model_validate(
        {
            "id": "graph",
            "created": 0,
            "owned_by": "langgraph-openai-serve",
            "lgos": {
                "description": "An LGOS graph",
                "features": ["interrupts"],
                "client_settings": {
                    "json_schema": {"type": "object"},
                    "defaults": {"style": "brief"},
                },
            },
        }
    )


def _source(model: ModelDetails, *, available: bool = True) -> httpx2.Client:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if not available:
            return httpx2.Response(503)
        if request.url.path == "/v1/models":
            return httpx2.Response(
                200, json=ModelList(data=[model]).model_dump(mode="json")
            )
        assert request.url.path == "/v1/models/graph"
        return httpx2.Response(200, json=model.model_dump(mode="json"))

    return httpx2.Client(
        base_url="https://source.invalid/v1/", transport=httpx2.MockTransport(respond)
    )


@pytest.fixture
def sources(model: ModelDetails) -> Iterator[dict[str, httpx2.Client]]:
    with _source(model) as team_a, _source(model) as team_b:
        yield {"team-a": team_a, "team-b": team_b}


def test_prepare_writes_zero_priced_rows_with_complete_metadata(
    sources: dict[str, httpx2.Client], model: ModelDetails, tmp_path: Path
) -> None:
    prepare_catalog(sources, tmp_path)

    pricing = json.loads((tmp_path / "pricing.json").read_text())
    assert pricing == {
        f"{provider}/graph": {
            "provider": provider,
            "mode": "responses",
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
        for provider in ("team-a", "team-b")
    }
    attributes = json.loads((tmp_path / "attributes.json").read_text())
    assert [(entry["provider"], entry["model"]) for entry in attributes] == [
        ("team-a", "graph"),
        ("team-b", "graph"),
    ]
    for entry in attributes:
        published = entry["additional_attributes"]
        assert published["description"] == "An LGOS graph"
        assert LangGraphModelExtension.model_validate_json(published["lgos"]) == (
            model.lgos
        )


def test_prepare_keeps_the_previous_catalog_when_a_source_fails(
    model: ModelDetails, tmp_path: Path
) -> None:
    (tmp_path / "pricing.json").write_text("previous")
    (tmp_path / "attributes.json").write_text("previous")

    with (
        _source(model) as healthy,
        _source(model, available=False) as failing,
        pytest.raises(httpx2.HTTPStatusError),
    ):
        prepare_catalog({"team-a": healthy, "team-b": failing}, tmp_path)

    assert (tmp_path / "pricing.json").read_text() == "previous"
    assert (tmp_path / "attributes.json").read_text() == "previous"


def test_sync_loads_new_rows_before_replacing_attributes() -> None:
    attributes = [
        ModelAttributes(
            provider=provider,
            model=name,
            additional_attributes={"description": "Graph", "lgos": "{}"},
        )
        for provider, name in (
            ("team-b", "chat"),
            ("team-a", "chat"),
            ("team-b", "research"),
        )
    ]
    requests: list[tuple[str, str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        return httpx2.Response(200, json={})

    with httpx2.Client(
        base_url="https://gateway.invalid/", transport=httpx2.MockTransport(respond)
    ) as gateway:
        sync_catalog(gateway, attributes)

    # The attribute write fails atomically unless every pricing row exists.
    assert requests == [
        ("POST", "/api/pricing/force-sync", None),
        ("POST", "/api/providers/team-a/refresh-models", None),
        ("POST", "/api/providers/team-b/refresh-models", None),
        (
            "PUT",
            "/api/models/catalog",
            [entry.model_dump() for entry in attributes],
        ),
    ]
