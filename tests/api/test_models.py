from typing import Literal

from openai import AsyncOpenAI
from pydantic import Field

from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    GraphRequest,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.protocol import JSON_SCHEMA_DIALECT, SETTINGS_METADATA_KEY
from tests.graph.support.interrupt import make_interrupt_graph
from tests.graph.support.message import make_message_graph

CLIENT_SETTINGS_SCHEMA_VERSION = 1


class PublicSettings(ClientSettings):
    enabled: bool = Field(default=True, title="Enabled")
    mode: Literal["brief", "detailed"] = "brief"


def bind_public_settings(graph_registry: GraphRegistry) -> GraphConfig:
    graph_config = GraphConfig(
        graph=make_message_graph(context_schema=PublicSettings),
        description="DUMMY",
        client_settings=PublicSettings,
    )
    graph_registry.register("test", graph_config)
    return graph_config


async def test_registered_graphs_are_exposed_with_standard_model_fields(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert response.data[0].id == "test"
    assert response.data[0].object == "model"
    assert response.data[0].owned_by == "langgraph-openai-serve"


async def test_retrieved_model_exposes_public_schema_and_defaults(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    bind_public_settings(graph_registry)

    response = await openai_client.models.retrieve("test")

    extension = (response.model_extra or {})["lgos"]
    client_settings = extension["client_settings"]
    assert client_settings["schema_version"] == CLIENT_SETTINGS_SCHEMA_VERSION
    assert client_settings["json_schema"]["$schema"] == JSON_SCHEMA_DIALECT
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


async def test_retrieved_model_exposes_sorted_graph_features(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    sqlite_checkpointer,
) -> None:
    graph_registry.register(
        "test",
        GraphConfig(
            graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
            description="DUMMY",
            features={
                GraphFeature.INTERRUPTS,
                GraphFeature.CLIENT_EVENTS,
                GraphFeature.FILE_INPUTS,
                GraphFeature.MCP_TOOLS,
            },
            run_coordinator=InMemoryRunCoordinator(),
        ),
    )

    response = await openai_client.models.retrieve("test")
    listed = await openai_client.models.list()

    extension = (response.model_extra or {})["lgos"]
    expected_extension = {
        "schema_version": 1,
        "description": "DUMMY",
        "features": ["client_events", "file_inputs", "interrupts", "mcp_tools"],
    }
    assert extension == expected_extension
    assert (listed.data[0].model_extra or {})["lgos"] == (expected_extension)


async def test_bound_client_settings_builds_validated_runtime_context(
    graph_registry: GraphRegistry,
) -> None:
    graph_config = bind_public_settings(graph_registry)
    request = GraphRequest(
        model="test",
        metadata={SETTINGS_METADATA_KEY: '{"enabled":false,"mode":"detailed"}'},
        user=None,
        tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
    )

    context = await graph_config.build_context(
        request,
        await graph_config.resolve_graph(),
    )

    assert context == PublicSettings(
        enabled=False,
        mode="detailed",
    )
