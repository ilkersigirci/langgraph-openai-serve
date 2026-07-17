from typing import Literal

import pytest
from openai import AsyncOpenAI, BadRequestError
from pydantic import ConfigDict, Field

from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphRegistry,
)
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.graph.client_settings import RUNTIME_SETTINGS_METADATA_KEY
from tests.graph.support.message import make_message_graph

pytestmark = pytest.mark.anyio
CLIENT_SETTINGS_SCHEMA_VERSION = 1


class PublicSettings(ClientSettings):
    enabled: bool = Field(default=True, title="Enabled")
    mode: Literal["brief", "detailed"] = "brief"


def bind_public_settings(graph_registry: GraphRegistry) -> GraphConfig:
    graph_config = GraphConfig(
        graph=make_message_graph(context_schema=PublicSettings),
        streamable_node_names=["generate"],
        client_settings=PublicSettings,
    )
    graph_registry.register("test", graph_config)
    return graph_config


async def test_registered_graphs_are_exposed_as_standard_model_summaries(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert response.data[0].id == "test"
    assert response.data[0].object == "model"
    assert response.data[0].owned_by == "langgraph-openai-serve"
    assert not response.data[0].model_extra


async def test_retrieved_model_exposes_public_schema_and_defaults(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    bind_public_settings(graph_registry)

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


async def test_model_retrieval_reuses_the_registration_schema(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    calls = 0

    def add_generation(schema: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        schema["generation"] = calls

    class StatefulSchemaSettings(ClientSettings):
        model_config = ConfigDict(json_schema_extra=add_generation)

        enabled: bool = True

    graph_registry.register(
        "stateful",
        GraphConfig(
            graph=make_message_graph(context_schema=StatefulSchemaSettings),
            client_settings=StatefulSchemaSettings,
        ),
    )

    first = await openai_client.models.retrieve("stateful")
    second = await openai_client.models.retrieve("stateful")

    first_extension = (first.model_extra or {})["langgraph_openai_serve"]
    second_extension = (second.model_extra or {})["langgraph_openai_serve"]
    assert first_extension["client_settings"]["json_schema"]["generation"] == 1
    assert second_extension["client_settings"]["json_schema"]["generation"] == 1
    assert calls == 1


async def test_bound_client_settings_builds_validated_runtime_context(
    graph_registry: GraphRegistry,
) -> None:
    graph_config = bind_public_settings(graph_registry)
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


async def test_bound_client_settings_does_not_coerce_json_values(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    bind_public_settings(graph_registry)

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
