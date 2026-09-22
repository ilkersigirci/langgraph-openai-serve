from typing import cast

import pytest
from pydantic import ValidationError

from langgraph_openai_serve import (
    BackgroundPolicy,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
)
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from langgraph_openai_serve.graph.graph_registry import GraphConfigurationError

EXPECTED_FACTORY_RESOLUTIONS = 2


def test_graph_config_is_immutable_and_copies_owned_collections(message_graph) -> None:
    features = {GraphFeature.CLIENT_EVENTS}
    server_tools = {"package_version"}
    config = GraphConfig(
        graph=message_graph,
        description="DUMMY",
        features=features,
        server_tools=server_tools,
    )

    features.clear()
    server_tools.clear()

    assert config.features == frozenset({GraphFeature.CLIENT_EVENTS})
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


def test_background_feature_must_be_derived_from_policy(message_graph) -> None:
    with pytest.raises(ValidationError, match=r"GraphConfig\.background"):
        GraphConfig(
            graph=message_graph,
            description="DUMMY",
            features={GraphFeature.BACKGROUND},
        )


def test_background_policy_requires_coordinator_and_excludes_interrupts(
    message_graph,
) -> None:
    with pytest.raises(ValidationError, match="run_coordinator"):
        GraphConfig(
            graph=message_graph,
            description="DUMMY",
            background=BackgroundPolicy(version="v1"),
        )

    with pytest.raises(ValidationError, match="does not support interrupt"):
        GraphConfig(
            graph=message_graph,
            description="DUMMY",
            features={GraphFeature.INTERRUPTS},
            background=BackgroundPolicy(version="v1"),
            run_coordinator=InMemoryRunCoordinator(),
        )


async def test_background_graph_requires_persistent_async_checkpointer(
    message_graph,
) -> None:
    config = GraphConfig(
        graph=message_graph,
        description="DUMMY",
        background=BackgroundPolicy(version="v1"),
        run_coordinator=InMemoryRunCoordinator(),
    )

    with pytest.raises(GraphConfigurationError, match="checkpointer"):
        await config.resolve_graph()


def test_graph_registry_requires_at_least_one_graph() -> None:
    with pytest.raises(ValueError, match="at least one graph"):
        GraphRegistry(registry={})


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

    with pytest.raises(ValidationError):
        GraphRegistry(registry={model_id: config})


def test_registry_copies_input_and_exposes_a_read_only_live_view(message_graph) -> None:
    config = GraphConfig(graph=message_graph, description="DUMMY")
    source = {"first": config}
    registry = GraphRegistry(registry=source)
    public_view = registry.registry

    source["outside"] = config
    registry.register("second", config)

    assert list(public_view) == ["first", "second"]
    assert "outside" not in public_view
    mutable_view = cast("dict[str, GraphConfig]", public_view)
    with pytest.raises(TypeError):
        mutable_view["third"] = config


def test_register_validates_before_mutation_and_preserves_order(message_graph) -> None:
    first = GraphConfig(graph=message_graph, description="First")
    second = GraphConfig(graph=message_graph, description="Second")
    registry = GraphRegistry(registry={"first": first, "second": second})
    replacement = GraphConfig(graph=message_graph, description="Replacement")

    with pytest.raises(ValidationError):
        registry.register("invalid/model", replacement)
    assert registry.get_graph_names() == ["first", "second"]
    assert registry.get_graph("first") is first

    with pytest.raises(TypeError, match="GraphConfig"):
        registry.register("third", cast("GraphConfig", object()))
    assert registry.get_graph_names() == ["first", "second"]
    assert registry.get_graph("first") is first

    registry.register("first", replacement)
    registry.register("third", second)

    assert registry.get_graph_names() == ["first", "second", "third"]
    assert registry.get_graph("first") is replacement


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

    with pytest.raises(GraphConfigurationError, match="compiled LangGraph"):
        await config.resolve_graph()
