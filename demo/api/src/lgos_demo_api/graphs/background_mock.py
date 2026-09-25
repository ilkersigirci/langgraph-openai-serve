"""
Deterministic graph that shows background execution working.

It calls no model, so the demo needs no provider to exercise queueing,
polling, and cancellation.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph_openai_serve import GraphConfig, GraphFeature

from lgos_demo_api.graphs.background_report import (
    BackgroundReportSettings,
    BackgroundReportState,
    prepare_report,
)


async def write_report(
    state: BackgroundReportState,
    runtime: Runtime[BackgroundReportSettings],
) -> dict[str, list[AIMessage]]:
    """Reply with a fixed report after the configured delay."""
    report = await prepare_report(state.messages[-1].text, runtime.context)
    return {"messages": [AIMessage(content=report)]}


workflow = StateGraph(BackgroundReportState, context_schema=BackgroundReportSettings)
workflow.add_node("write_report", write_report)
workflow.add_edge(START, "write_report")
workflow.add_edge("write_report", END)

background_mock_graph = workflow.compile()

background_mock_graph_config = GraphConfig(
    graph=background_mock_graph,
    description=(
        "Demonstrates background execution with a deterministic, delayed "
        "mock report that calls no model."
    ),
    features={GraphFeature.BACKGROUND},
    client_settings=BackgroundReportSettings,
)

__all__ = [
    "background_mock_graph",
    "background_mock_graph_config",
]
