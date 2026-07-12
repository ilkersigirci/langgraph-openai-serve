"""Coordinator subgraph for the complex demo."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from demo.api.graphs.subgraphs.api_contract import create_api_contract_graph
from demo.api.graphs.subgraphs.docs import create_docs_graph
from demo.api.graphs.subgraphs.schemas import (
    ComplexSubgraphState,
    Route,
)


def _select_route(question: str) -> Route:
    normalized = question.lower()
    api_words = ("api", "openai", "adapter", "stream", "serve")
    if any(word in normalized for word in api_words):
        return "api"
    return "docs"


async def route_question(state: ComplexSubgraphState) -> dict[str, str]:
    normalized_question = " ".join(state.question.strip().split())
    route = _select_route(normalized_question)
    return {
        "normalized_question": normalized_question,
        "route": route,
    }


def select_subgraph(state: ComplexSubgraphState) -> str:
    if state.route == "api":
        return "api_contract_graph"
    return "docs_graph"


def create_specialist_team_graph(
    api_contract_graph: CompiledStateGraph | None = None,
    docs_graph: CompiledStateGraph | None = None,
) -> CompiledStateGraph:
    """Create the coordinator graph with specialist subgraphs as direct nodes."""

    api_contract_subgraph = api_contract_graph or create_api_contract_graph()
    docs_subgraph = docs_graph or create_docs_graph()

    builder = StateGraph(ComplexSubgraphState)
    builder.add_node("route_question", route_question)
    builder.add_node("api_contract_graph", api_contract_subgraph)
    builder.add_node("docs_graph", docs_subgraph)
    builder.add_edge(START, "route_question")
    builder.add_conditional_edges(
        "route_question",
        select_subgraph,
        ["api_contract_graph", "docs_graph"],
    )
    builder.add_edge("api_contract_graph", END)
    builder.add_edge("docs_graph", END)
    return builder.compile()
