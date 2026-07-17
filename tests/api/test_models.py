from typing import Literal

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from openai import AsyncOpenAI, BadRequestError
from pydantic import Field, ValidationError

from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
)
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.api.models.schemas import ModelClientSettings
from langgraph_openai_serve.graph.client_settings import RUNTIME_SETTINGS_METADATA_KEY
from tests.graph.support.interrupt import make_interrupt_graph

pytestmark = pytest.mark.anyio
CLIENT_SETTINGS_SCHEMA_VERSION = 1


class PublicSettings(ClientSettings):
    enabled: bool = Field(default=True, title="Enabled")
    mode: Literal["brief", "detailed"] = "brief"


def test_client_settings_descriptor_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelClientSettings.model_validate(
            {
                "json_schema": {},
                "defaults": {},
                "unknown": "value",
            }
        )


async def test_registered_graphs_are_exposed_as_standard_model_summaries(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert response.data[0].id == "test"
    assert response.data[0].object == "model"
    assert response.data[0].owned_by == "langgraph-openai-serve"
    assert not response.data[0].model_extra


async def test_model_details_are_available_through_openai_client(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.retrieve("test")

    assert response.id == "test"
    extension = (response.model_extra or {})["langgraph_openai_serve"]
    assert extension == {"schema_version": 1, "features": []}


async def test_graph_features_are_available_only_on_model_retrieval(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    graph_registry.register(
        "test",
        GraphConfig(
            graph=make_interrupt_graph(checkpointer=InMemorySaver()),
            features={GraphFeature.INTERRUPTS},
        ),
    )

    listed = await openai_client.models.list()
    retrieved = await openai_client.models.retrieve("test")

    assert not listed.data[0].model_extra
    extension = (retrieved.model_extra or {})["langgraph_openai_serve"]
    assert extension == {"schema_version": 1, "features": ["interrupts"]}


async def test_model_discovery_does_not_resolve_lazy_graph_factories(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    message_graph,
) -> None:
    resolved = False

    def lazy_graph():
        nonlocal resolved
        resolved = True
        return message_graph

    graph_registry.register("lazy", GraphConfig(graph=lazy_graph))

    await openai_client.models.list()
    await openai_client.models.retrieve("lazy")

    assert resolved is False


async def test_retrieved_model_exposes_public_schema_and_defaults(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    graph_registry.registry["test"].client_settings = PublicSettings

    response = await openai_client.models.retrieve("test")

    extension = (response.model_extra or {})["langgraph_openai_serve"]
    client_settings = extension["client_settings"]
    assert client_settings["schema_version"] == CLIENT_SETTINGS_SCHEMA_VERSION
    assert client_settings["json_schema"]["additionalProperties"] is False
    assert client_settings["json_schema"]["properties"]["enabled"] == {
        "default": True,
        "title": "Enabled",
        "type": "boolean",
    }
    assert client_settings["defaults"] == {
        "enabled": True,
        "mode": "brief",
    }


async def test_bound_client_settings_builds_validated_runtime_context(
    graph_registry: GraphRegistry,
) -> None:
    graph_config = graph_registry.registry["test"]
    graph_config.client_settings = PublicSettings
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Hello"}],
        metadata={RUNTIME_SETTINGS_METADATA_KEY: '{"enabled":false,"mode":"detailed"}'},
    )

    context = await graph_config.build_context(
        request,
        await graph_config.resolve_graph(),
    )

    assert context == PublicSettings(
        enabled=False,
        mode="detailed",
    )


async def test_invalid_bound_client_settings_returns_openai_error(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    graph_registry.registry["test"].client_settings = PublicSettings

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={RUNTIME_SETTINGS_METADATA_KEY: "not-json"},
        )

    assert exc_info.value.response.json() == {
        "error": {
            "message": (
                "Invalid runtime settings: "
                "Invalid JSON: expected ident at line 1 column 2"
            ),
            "type": "invalid_request_error",
            "param": f"metadata.{RUNTIME_SETTINGS_METADATA_KEY}",
            "code": None,
        }
    }


async def test_bound_client_settings_does_not_coerce_json_values(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    graph_registry.registry["test"].client_settings = PublicSettings

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={RUNTIME_SETTINGS_METADATA_KEY: '{"enabled":"false"}'},
        )

    assert exc_info.value.response.json() == {
        "error": {
            "message": (
                "Invalid runtime setting for enabled: Input should be a valid boolean"
            ),
            "type": "invalid_request_error",
            "param": f"metadata.{RUNTIME_SETTINGS_METADATA_KEY}",
            "code": None,
        }
    }
