"""Keyword grandchild subgraph for the complex demo."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from demo.api.graphs.subgraphs.schemas import KeywordState
from langgraph_openai_serve.utils.fake_llm import stream_fake_chat_response


async def extract_keywords(state: KeywordState) -> dict[str, list[str]]:
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
    response = "Keyword subgraph: selected " + ", ".join(selected_keywords)
    await stream_fake_chat_response(f"{response}\n", source_text)
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
