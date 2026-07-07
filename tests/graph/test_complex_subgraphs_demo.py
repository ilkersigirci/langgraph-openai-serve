import pytest
from demo.api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from demo.api.graphs.subgraphs.docs import create_docs_graph
from demo.api.graphs.subgraphs.keyword import extract_keywords
from demo.api.graphs.subgraphs.schemas import (
    ApiContractState,
    ComplexSubgraphState,
    DocsState,
    KeywordState,
)
from demo.api.graphs.subgraphs.specialist_team import create_specialist_team_graph
from pydantic import BaseModel

from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream


@pytest.fixture
def complex_subgraphs_registry() -> GraphRegistry:
    return GraphRegistry(
        registry={"complex-subgraphs": create_complex_subgraphs_graph_config()},
    )


def test_complex_subgraphs_use_structural_pydantic_subgraphs() -> None:
    specialist_team_graph = create_specialist_team_graph()
    docs_graph = create_docs_graph()
    parent_node_names = {
        node.name
        for node in specialist_team_graph.get_graph(xray=False).nodes.values()
    }
    node_names = {
        node.name for node in specialist_team_graph.get_graph(xray=True).nodes.values()
    }

    assert "api_contract_graph" in parent_node_names
    assert "docs_graph" in parent_node_names
    assert "collect_contract_checks" in node_names
    assert "summarize_contract" in node_names
    assert "summarize_docs" in node_names
    assert "extract_keywords" in node_names

    docs_parent_node_names = {
        node.name for node in docs_graph.get_graph(xray=False).nodes.values()
    }
    docs_node_names = {
        node.name for node in docs_graph.get_graph(xray=True).nodes.values()
    }
    assert "keyword_graph" in docs_parent_node_names
    assert "extract_keywords" in docs_node_names

    states = (ComplexSubgraphState, ApiContractState, DocsState, KeywordState)
    assert all(issubclass(state, BaseModel) for state in states)


@pytest.mark.anyio
async def test_keyword_node_uses_general_fallback() -> None:
    result = await extract_keywords(KeywordState(normalized_question="Hello."))

    assert result == {
        "keywords": ["general"],
        "checks": ["nested keyword subgraph selected `general`"],
    }


@pytest.mark.anyio
async def test_complex_subgraphs_demo_routes_api_question_to_api_subgraph(
    make_request,
    complex_subgraphs_registry: GraphRegistry,
) -> None:
    request = make_request(
        "complex-subgraphs",
        content="Show OpenAI adapter streaming with nested subgraphs.",
        user="demo-user",
    )

    response, usage = await run_langgraph(
        "complex-subgraphs",
        request.messages,
        complex_subgraphs_registry,
        request,
    )

    assert "API contract:" in response
    assert "Docs specialist:" not in response
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0


@pytest.mark.anyio
async def test_complex_subgraphs_demo_routes_docs_question_to_docs_subgraph(
    make_request,
    complex_subgraphs_registry: GraphRegistry,
) -> None:
    request = make_request(
        "complex-subgraphs",
        content="Show nested subgraph routing docs.",
        user="demo-user",
    )

    response, usage = await run_langgraph(
        "complex-subgraphs",
        request.messages,
        complex_subgraphs_registry,
        request,
    )

    assert "Docs specialist:" in response
    assert "nested keyword subgraph selected" in response
    assert "API contract:" not in response
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0


@pytest.mark.anyio
async def test_complex_subgraphs_demo_streams_only_routed_docs_subgraph(
    make_request,
    complex_subgraphs_registry: GraphRegistry,
) -> None:
    request = make_request(
        "complex-subgraphs",
        content="Show nested subgraph routing docs.",
    )

    chunks = [
        event
        async for event in run_langgraph_stream(
            "complex-subgraphs",
            request.messages,
            complex_subgraphs_registry,
            request,
        )
    ]

    assert all(isinstance(chunk, str) for chunk in chunks)
    response = "".join(chunks)
    keyword_index = response.index("Keyword subgraph:")
    docs_index = response.index("Docs specialist:")
    assert keyword_index < docs_index
    assert "API contract:" not in response
