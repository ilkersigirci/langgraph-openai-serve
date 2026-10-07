from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from langgraph_openai_serve import (
    GraphConfig,
    GraphError,
    GraphFeature,
    GraphRegistry,
)
from tests.graph.support.schemas import MessageState

EXPECTED_FACTORY_RESOLUTIONS = 2


def test_graph_config_is_immutable_and_copies_owned_collections(message_graph) -> None:
    features = {GraphFeature.FILE_INPUTS}
    server_tools = {"package_version"}
    config = GraphConfig(
        graph=message_graph,
        description="DUMMY",
        features=features,
        server_tools=server_tools,
    )

    features.clear()
    server_tools.clear()

    assert config.features == frozenset({GraphFeature.FILE_INPUTS})
    assert config.server_tools == frozenset({"package_version"})
    with pytest.raises(ValidationError, match="frozen"):
        config.description = "Changed"


def test_graph_config_rejects_empty_server_tool_names(message_graph) -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        GraphConfig(
            graph=message_graph,
            description="DUMMY",
            server_tools={""},
        )


def test_graph_config_rejects_unknown_fields(message_graph) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GraphConfig.model_validate(
            {
                "graph": message_graph,
                "description": "DUMMY",
                "unknown": True,
            }
        )


def test_graph_registry_requires_at_least_one_graph() -> None:
    with pytest.raises(ValueError, match="at least one graph"):
        GraphRegistry(graphs={})


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("", id="empty"),
        pytest.param("group/model", id="slash"),
        pytest.param(".", id="current-path"),
        pytest.param("..", id="parent-path"),
    ],
)
def test_graph_registry_rejects_unaddressable_model_ids(
    message_graph,
    model_id: str,
) -> None:
    config = GraphConfig(graph=message_graph, description="DUMMY")

    with pytest.raises(ValueError, match="not addressable"):
        GraphRegistry(graphs={model_id: config})


async def test_graph_resolvers_preserve_their_lifetimes(message_graph) -> None:
    sync_calls = 0
    async_calls = 0

    def sync_factory():
        nonlocal sync_calls
        sync_calls += 1
        return message_graph

    async def async_factory():
        nonlocal async_calls
        async_calls += 1
        return message_graph

    direct = GraphConfig(graph=message_graph, description="Direct")
    sync = GraphConfig(graph=sync_factory, description="Sync")
    async_ = GraphConfig(graph=async_factory, description="Async")

    assert await direct.resolve_graph() is message_graph
    assert await direct.resolve_graph() is message_graph
    assert await sync.resolve_graph() is message_graph
    assert await sync.resolve_graph() is message_graph
    assert await async_.resolve_graph() is message_graph
    assert await async_.resolve_graph() is message_graph
    assert sync_calls == EXPECTED_FACTORY_RESOLUTIONS
    assert async_calls == EXPECTED_FACTORY_RESOLUTIONS


async def test_factory_result_must_be_a_compiled_state_graph() -> None:
    config = GraphConfig(graph=object, description="DUMMY")

    with pytest.raises(GraphError, match="compiled LangGraph"):
        await config.resolve_graph()


def two_step_graph(**compile_options):
    def step(_state: MessageState) -> dict:
        return {"messages": [AIMessage(content="step")]}

    graph = StateGraph(MessageState).add_node("first", step).add_node("second", step)
    graph = graph.set_entry_point("first").add_edge("first", "second")
    return graph.set_finish_point("second").compile(**compile_options)


def with_subgraph(subgraph):
    graph = StateGraph(MessageState).add_node("subgraph", subgraph)
    return graph.set_entry_point("subgraph").set_finish_point("subgraph").compile()


@pytest.mark.parametrize(
    ("build", "error"),
    [
        pytest.param(
            lambda: two_step_graph(interrupt_before=["second"]),
            "interrupt_before",
            id="interrupt-before",
        ),
        pytest.param(
            lambda: two_step_graph(interrupt_after=["first"]),
            "interrupt_after",
            id="interrupt-after",
        ),
        pytest.param(
            lambda: with_subgraph(two_step_graph(interrupt_after=["first"])),
            "interrupt_after",
            id="subgraph-interrupt-after",
        ),
        pytest.param(
            lambda: two_step_graph(checkpointer=InMemorySaver()),
            "checkpointer",
            id="checkpointer-without-interrupts",
        ),
        pytest.param(
            lambda: two_step_graph(checkpointer=True),
            "checkpointer",
            id="inherited-checkpointer",
        ),
    ],
)
async def test_graphs_lgos_cannot_serve_are_rejected(
    build: Callable[[], CompiledStateGraph], error: str
) -> None:
    graph = build()

    with pytest.raises(GraphError, match=error):
        GraphConfig(graph=graph, description="DUMMY")
    factory = GraphConfig(graph=lambda: graph, description="DUMMY")
    with pytest.raises(GraphError, match=error):
        await factory.resolve_graph()
