"""Docs specialist subgraph for the complex demo."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from demo.api.graphs.subgraphs.keyword import create_keyword_graph
from demo.api.graphs.subgraphs.schemas import DocsState
from langgraph_openai_serve.utils.fake_llm import stream_fake_chat_response


async def summarize_docs(state: DocsState) -> dict[str, str]:
    keyword_list = ", ".join(state.keywords)
    check_list = "; ".join(state.checks)
    summary = (
        f"Docs specialist: covered {keyword_list}; {check_list}; "
        "keyword subgraph shared the docs state channels"
    )
    prompt = "\n".join([state.normalized_question or state.question, check_list])
    streamed_summary = await stream_fake_chat_response(f"{summary}\n", prompt)
    return {"answer": streamed_summary.removesuffix("\n")}


def create_docs_graph(
    keyword_graph: CompiledStateGraph | None = None,
) -> CompiledStateGraph:
    """Create the docs specialist graph."""

    keyword_subgraph = keyword_graph or create_keyword_graph()

    builder = StateGraph(DocsState)
    builder.add_node("keyword_graph", keyword_subgraph)
    builder.add_node("summarize_docs", summarize_docs)
    builder.add_edge(START, "keyword_graph")
    builder.add_edge("keyword_graph", "summarize_docs")
    builder.add_edge("summarize_docs", END)
    return builder.compile()
