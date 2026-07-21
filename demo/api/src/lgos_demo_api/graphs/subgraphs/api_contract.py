"""API contract specialist subgraph for the complex demo."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve.utils.fake_llm import stream_fake_chat_response

from lgos_demo_api.graphs.subgraphs.schemas import ApiContractState


async def collect_contract_checks(state: ApiContractState) -> dict[str, Any]:
    source_text = state.normalized_question or state.question
    normalized = " ".join(source_text.lower().split())
    checks = [
        "OpenAI chat messages were adapted into native graph input",
        "native graph output is rendered back as assistant text",
    ]
    if "stream" in normalized:
        checks.append("streamable nested node names can be exposed safely")
    return {"checks": checks}


async def summarize_contract(state: ApiContractState) -> dict[str, str]:
    summary = "API contract: " + "; ".join(state.checks)
    prompt = "\n".join(state.checks)
    streamed_summary = await stream_fake_chat_response(f"{summary}\n", prompt)
    return {
        "answer": streamed_summary.removesuffix("\n"),
    }


def create_api_contract_graph() -> CompiledStateGraph:
    """Create the API contract specialist graph."""

    builder = StateGraph(ApiContractState)
    builder.add_node("collect_contract_checks", collect_contract_checks)
    builder.add_node("summarize_contract", summarize_contract)
    builder.add_edge(START, "collect_contract_checks")
    builder.add_edge("collect_contract_checks", "summarize_contract")
    builder.add_edge("summarize_contract", END)
    return builder.compile()
