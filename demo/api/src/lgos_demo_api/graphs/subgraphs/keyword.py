"""Keyword grandchild subgraph for the complex demo."""

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve import status_event

from lgos_demo_api.graphs.subgraphs.schemas import KeywordState


def extract_keywords(state: KeywordState) -> dict[str, list[str]]:
    source_text = state.normalized_question or state.question
    normalized = source_text.lower()
    candidates = (
        "subgraph",
        "streaming",
        "adapter",
        "openai",
        "context",
        "routing",
    )
    keywords = [candidate for candidate in candidates if candidate in normalized]
    selected_keywords = keywords or ["general"]
    get_stream_writer()(
        status_event(
            f"Selected keywords: {', '.join(selected_keywords)}",
            done=True,
            namespace=("docs", "keywords"),
        )
    )
    return {
        "keywords": selected_keywords,
        "checks": [
            "nested keyword subgraph selected "
            + ", ".join(f"`{keyword}`" for keyword in selected_keywords)
        ],
    }


def create_keyword_graph() -> CompiledStateGraph:
    """Create the keyword extraction subgraph."""
    return (
        StateGraph(KeywordState)
        .add_node("extract_keywords", extract_keywords)
        .add_edge(START, "extract_keywords")
        .add_edge("extract_keywords", END)
        .compile()
    )
