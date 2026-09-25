"""Deterministic background report with durable approval and resumption."""

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from langgraph_openai_serve import GraphConfig, GraphFeature
from langgraph_openai_serve.graph.interrupt import RunCoordinator

from lgos_demo_api.graphs.background_report import (
    BackgroundReportSettings,
    BackgroundReportState,
    prepare_report,
    wait_for_report,
)


class BackgroundInterruptState(BackgroundReportState):
    """Keep the prepared report across the review pause."""

    report: str = ""
    decision: Literal["approve", "reject"] | None = None


BackgroundInterruptGraph = CompiledStateGraph[
    BackgroundInterruptState,
    BackgroundReportSettings,
    BackgroundInterruptState,
    BackgroundInterruptState,
]


async def prepare_for_review(
    state: BackgroundInterruptState,
    runtime: Runtime[BackgroundReportSettings],
) -> dict[str, str]:
    return {"report": await prepare_report(state.messages[-1].text, runtime.context)}


def review_report(
    state: BackgroundInterruptState,
) -> dict[str, Literal["approve", "reject"]]:
    # Resumption restarts this node; preparation belongs in the previous node.
    answer = interrupt(
        {
            "question": "Approve this report?",
            "report": state.report,
            "choices": ["approve", "reject"],
            "allow_other": False,
        }
    )
    if isinstance(answer, str):
        decision = answer.strip().lower()
        if decision in {"approve", "reject"}:
            return {"decision": decision}
    raise ValueError("Report review response must be approve or reject.")


async def finish_report(
    state: BackgroundInterruptState,
    runtime: Runtime[BackgroundReportSettings],
) -> dict[str, list[AIMessage]]:
    if state.decision == "approve":
        await wait_for_report(runtime.context)
        content = state.report
    else:
        content = "Report rejected; no report finalized."
    return {"messages": [AIMessage(content=content)]}


def create_background_interrupt_graph(
    checkpointer: BaseCheckpointSaver,
) -> BackgroundInterruptGraph:
    workflow = StateGraph(
        BackgroundInterruptState, context_schema=BackgroundReportSettings
    )
    workflow.add_node("prepare_report", prepare_for_review)
    workflow.add_node("review_report", review_report)
    workflow.add_node("finish_report", finish_report)
    workflow.add_edge(START, "prepare_report")
    workflow.add_edge("prepare_report", "review_report")
    workflow.add_edge("review_report", "finish_report")
    workflow.add_edge("finish_report", END)
    return workflow.compile(checkpointer=checkpointer)


def create_background_interrupt_graph_config(
    graph_factory: Callable[[], BackgroundInterruptGraph],
    run_coordinator: RunCoordinator,
) -> GraphConfig:
    return GraphConfig(
        graph=graph_factory,
        description=(
            "Demonstrates deterministic background preparation, human approval, "
            "and background resumption without a model call."
        ),
        features={GraphFeature.BACKGROUND, GraphFeature.INTERRUPTS},
        client_settings=BackgroundReportSettings,
        run_coordinator=run_coordinator,
    )


__all__ = [
    "create_background_interrupt_graph",
    "create_background_interrupt_graph_config",
]
