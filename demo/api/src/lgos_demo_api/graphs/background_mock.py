"""
Deterministic graph that shows background execution working.

It calls no model, so the demo needs no provider to exercise queueing,
polling, and cancellation.
"""

from collections.abc import Sequence
from typing import Annotated

from anyio import sleep
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph_openai_serve import ClientSettings, GraphConfig, GraphFeature
from pydantic import BaseModel, Field


class BackgroundMockState(BaseModel):
    """Transcript that receives the report."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class BackgroundMockSettings(ClientSettings):
    """Per-request settings for the mock response."""

    delay_seconds: int = Field(
        default=5,
        ge=0,
        le=300,
        title="Delay (seconds)",
        description="Time the report takes, leaving room to poll or cancel it.",
    )


async def write_report(
    state: BackgroundMockState,
    runtime: Runtime[BackgroundMockSettings],
) -> dict[str, list[AIMessage]]:
    """Reply with a fixed report after the configured delay."""
    report_settings = runtime.context or BackgroundMockSettings()
    await sleep(report_settings.delay_seconds)
    request = state.messages[-1].text
    return {"messages": [AIMessage(content=f"Background report for: {request}")]}


workflow = StateGraph(BackgroundMockState, context_schema=BackgroundMockSettings)
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
    client_settings=BackgroundMockSettings,
)

__all__ = [
    "BackgroundMockSettings",
    "background_mock_graph",
    "background_mock_graph_config",
]
