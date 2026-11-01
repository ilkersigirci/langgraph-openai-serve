from typing import Literal

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from openai import AsyncOpenAI
from pydantic import Field

from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
)
from langgraph_openai_serve.protocol import JSON_SCHEMA_DIALECT, SETTINGS_METADATA_KEY
from tests.graph.support.interrupt import make_interrupt_graph
from tests.graph.support.message import make_message_graph
from tests.graph.support.schemas import MessageState


class PublicSettings(ClientSettings):
    enabled: bool = Field(default=True, title="Enabled")
    mode: Literal["brief", "detailed"] = "brief"


def bind_public_settings(graph_registry: GraphRegistry) -> GraphConfig:
    graph_config = GraphConfig(
        graph=make_message_graph(context_schema=PublicSettings),
        description="DUMMY",
        client_settings=PublicSettings,
    )
    graph_registry.graphs["test"] = graph_config
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
    listed = await openai_client.models.list()

    # Only retrieval carries the settings descriptor.
    assert [
        "client_settings" in (model.model_extra or {})["lgos"] for model in listed.data
    ] == [False]
    extension = (response.model_extra or {})["lgos"]
    client_settings = extension["client_settings"]
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


async def test_models_advertise_sorted_features_and_tool_names(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    sqlite_checkpointer,
) -> None:
    graph_registry.graphs["test"] = GraphConfig(
        graph=make_interrupt_graph(checkpointer=sqlite_checkpointer),
        description="DUMMY",
        features={
            GraphFeature.INTERRUPTS,
            GraphFeature.FILE_INPUTS,
            GraphFeature.MCP_TOOLS,
        },
        server_tools={"web_search", "package_version"},
        client_tools={"display_file"},
    )

    response = await openai_client.models.retrieve("test")
    listed = await openai_client.models.list()

    extension = (response.model_extra or {})["lgos"]
    expected_extension = {
        "description": "DUMMY",
        "features": ["file_inputs", "interrupts", "mcp_tools"],
        "server_tools": ["package_version", "web_search"],
        "client_tools": ["display_file"],
    }
    assert extension == expected_extension
    assert (listed.data[0].model_extra or {})["lgos"] == (expected_extension)


@pytest.mark.parametrize("api", ["responses", "chat"])
async def test_request_settings_become_the_runtime_context(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    api: str,
) -> None:
    def echo_context(
        _state: MessageState, runtime: Runtime[PublicSettings]
    ) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content=runtime.context.model_dump_json())]}

    graph_registry.graphs["test"] = GraphConfig(
        graph=StateGraph(MessageState, context_schema=PublicSettings)
        .add_node("echo", echo_context)
        .set_entry_point("echo")
        .set_finish_point("echo")
        .compile(),
        description="DUMMY",
        client_settings=PublicSettings,
    )
    metadata = {SETTINGS_METADATA_KEY: '{"enabled":false}'}

    if api == "responses":
        response = await openai_client.responses.create(
            model="test", input="Hi", metadata=metadata
        )
        context = response.output_text
    else:
        completion = await openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            metadata=metadata,
        )
        context = completion.choices[0].message.content or ""

    # Omitted settings keep their advertised defaults.
    assert PublicSettings.model_validate_json(context) == PublicSettings(
        enabled=False, mode="brief"
    )
