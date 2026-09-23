"""Checkpointed model-backed report agent for the background Responses demo."""

from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from anyio import sleep
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph_openai_serve import ClientSettings, GraphConfig, GraphFeature
from langgraph_openai_serve.graph.coordination import RunCoordinator
from pydantic import Field

from lgos_demo_api.core.settings import settings


class BackgroundReportState(TypedDict, total=False):
    """Durable input, intermediate draft, and final transcript."""

    messages: Annotated[list[BaseMessage], add_messages]
    draft: AIMessage


class BackgroundReportSettings(ClientSettings):
    """Per-request settings; retries reuse them from the persisted request."""

    finalize_delay_seconds: int = Field(
        default=5,
        ge=0,
        le=300,
        title="Finalize delay (seconds)",
        description=(
            "Pause after the draft checkpoint before publishing, leaving time "
            "to stop the worker and watch the run resume from the checkpoint."
        ),
    )


BackgroundReportGraph = CompiledStateGraph[Any, BackgroundReportSettings, Any, Any]


async def draft_report(state: BackgroundReportState) -> dict[str, AIMessage]:
    """Generate the costly draft that should survive a worker restart."""
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,
    )
    draft = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Prepare a concise report with a title, findings, and next "
                    "actions. State assumptions instead of inventing sources."
                )
            ),
            *state["messages"],
        ]
    )
    return {"draft": draft}


async def publish_report(
    state: BackgroundReportState,
    runtime: Runtime[BackgroundReportSettings],
) -> dict[str, list[AIMessage]]:
    """Publish the durable draft after a visible crash-recovery window."""
    report_settings = runtime.context or BackgroundReportSettings()
    await sleep(report_settings.finalize_delay_seconds)
    return {"messages": [state["draft"]]}


def create_background_report_graph(
    checkpointer: BaseCheckpointSaver,
) -> BackgroundReportGraph:
    """Compile a two-boundary report agent with persistent checkpoints."""
    workflow = StateGraph(
        BackgroundReportState,  # ty: ignore[invalid-argument-type]  # LangGraph supports TypedDict state at runtime.
        context_schema=BackgroundReportSettings,
    )
    workflow.add_node("draft_report", draft_report)
    workflow.add_node("publish_report", publish_report)
    workflow.add_edge(START, "draft_report")
    workflow.add_edge("draft_report", "publish_report")
    workflow.add_edge("publish_report", END)
    return workflow.compile(checkpointer=checkpointer)


def create_background_report_config(
    graph_factory: Callable[[], BackgroundReportGraph],
    run_coordinator: RunCoordinator,
) -> GraphConfig:
    """Declare the background graph shared by the API and Hatchet worker."""
    return GraphConfig(
        graph=graph_factory,
        description=(
            "Creates a checkpointed report in an independently deployed "
            "background worker."
        ),
        run_coordinator=run_coordinator,
        features={GraphFeature.BACKGROUND},
        client_settings=BackgroundReportSettings,
    )


__all__ = [
    "BackgroundReportSettings",
    "BackgroundReportState",
    "create_background_report_config",
    "create_background_report_graph",
]
