import pytest
from langgraph.checkpoint.memory import InMemorySaver
from openai import AsyncOpenAI

from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRegistry
from tests.graph.support.interrupt import make_interrupt_graph

pytestmark = pytest.mark.anyio


async def test_registered_graphs_are_exposed_as_models(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert response.data[0].id == "test"
    assert response.data[0].object == "model"
    extension = (response.data[0].model_extra or {})["langgraph_openai_serve"]
    assert extension == {"schema_version": 1, "features": []}


async def test_graph_features_are_available_through_openai_client(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    graph_registry.registry["test"] = GraphConfig(
        graph=make_interrupt_graph(checkpointer=InMemorySaver()),
        features={GraphFeature.INTERRUPTS},
    )

    response = await openai_client.models.list()

    extension = (response.data[0].model_extra or {})["langgraph_openai_serve"]
    assert extension == {"schema_version": 1, "features": ["interrupts"]}
