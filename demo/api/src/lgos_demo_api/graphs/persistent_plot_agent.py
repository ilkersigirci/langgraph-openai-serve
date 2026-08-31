"""A persistent chart managed by a LangChain agent."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Any, Literal

from fastapi import status
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphFeature,
    client_event,
)
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from openai.types.shared import ErrorObject
from pydantic import BaseModel, ConfigDict, Field

from lgos_demo_api.settings import settings

ARTIFACT_KEY = "quarterly-revenue"
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
Quarter = Literal["Q1", "Q2", "Q3", "Q4"]
SYSTEM_PROMPT = """You manage one persistent quarterly revenue chart.

For reads or display requests, use show_quarterly_revenue. For edits, call
update_quarterly_revenue directly; it reads the current values itself. Put every
edit requested in one update call, and never call both tools for the same
request. Never invent stored values. After a tool call, answer concisely and
mention the important result.
"""


class PlotDocument(BaseModel):
    """Canonical chart data stored independently of either UI."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1] = 1
    q1: float = Field(default=120, ge=0)
    q2: float = Field(default=180, ge=0)
    q3: float = Field(default=150, ge=0)
    q4: float = Field(default=230, ge=0)


class RevenueUpdate(BaseModel):
    """One validated chart edit selected by the agent."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    quarter: Quarter
    revenue: float = Field(ge=0, description="Revenue in thousands")


class PersistentPlotAgentSettings(ClientSettings):
    """Small presentation settings selected by the OpenAI client."""

    chart_type: Literal["bar", "line"] = Field(
        default="bar",
        title="Chart type",
        description="Render the stored values as a bar or line chart.",
    )
    currency: Literal["USD", "EUR"] = Field(
        default="USD",
        title="Currency",
        description="Label the stored revenue values as USD or EUR.",
    )
    show_legend: bool = Field(
        default=True,
        title="Show legend",
        description="Display the chart legend.",
    )


class ChartSeries(BaseModel):
    """One portable series in a chart artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    name: str = Field(min_length=1)
    values: list[float]


class ChartArtifact(BaseModel):
    """A small UI-neutral chart snapshot for rich clients."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    kind: Literal["chart"] = "chart"
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    chart_type: Literal["bar", "line"]
    labels: list[str]
    series: list[ChartSeries]
    x_axis_title: str = Field(min_length=1)
    y_axis_title: str = Field(min_length=1)
    show_legend: bool


@dataclass(frozen=True, slots=True)
class PersistentPlotAgentContext:
    user_id: str
    session_id: str
    settings: PersistentPlotAgentSettings


PersistentPlotAgent = CompiledStateGraph[Any, PersistentPlotAgentContext, Any, Any]


def _currency_symbol(currency: Literal["USD", "EUR"]) -> str:
    return "$" if currency == "USD" else "€"


def _values(document: PlotDocument) -> list[float]:
    return [document.q1, document.q2, document.q3, document.q4]


def _summary(
    document: PlotDocument,
    settings: PersistentPlotAgentSettings,
) -> str:
    values = _values(document)
    highest_index = values.index(max(values))
    symbol = _currency_symbol(settings.currency)
    return f"{QUARTERS[highest_index]} is highest at {symbol}{max(values):g}k."


def _artifact(
    document: PlotDocument,
    settings: PersistentPlotAgentSettings,
) -> ChartArtifact:
    return ChartArtifact(
        id=ARTIFACT_KEY,
        title="Quarterly revenue",
        summary=_summary(document, settings),
        chart_type=settings.chart_type,
        labels=list(QUARTERS),
        series=[ChartSeries(name="Revenue", values=_values(document))],
        x_axis_title="Quarter",
        y_axis_title=f"Revenue ({settings.currency}, thousands)",
        show_legend=settings.show_legend,
    )


def _thread_scope(user_id: str, session_id: str) -> str:
    return sha256(f"{user_id}\0{session_id}".encode()).hexdigest()


def _thread_namespace(context: PersistentPlotAgentContext) -> tuple[str, ...]:
    return (
        "demo",
        "persistent-plot-agent",
        "threads",
        _thread_scope(context.user_id, context.session_id),
    )


def _runtime_dependencies(
    runtime: ToolRuntime[PersistentPlotAgentContext],
) -> tuple[BaseStore, PersistentPlotAgentContext]:
    if runtime.store is None:
        msg = "The persistent plot agent requires a LangGraph store."
        raise RuntimeError(msg)
    if runtime.context is None:
        msg = "The persistent plot agent requires runtime context."
        raise RuntimeError(msg)
    return runtime.store, runtime.context


async def _load_document(
    store: BaseStore,
    context: PersistentPlotAgentContext,
) -> PlotDocument:
    item = await store.aget(_thread_namespace(context), ARTIFACT_KEY)
    return (
        PlotDocument.model_validate(item.value) if item is not None else PlotDocument()
    )


def _publish(
    document: PlotDocument,
    context: PersistentPlotAgentContext,
    runtime: ToolRuntime[PersistentPlotAgentContext],
) -> ChartArtifact:
    artifact = _artifact(document, context.settings)
    runtime.stream_writer(
        client_event(
            "artifact",
            artifact.model_dump(mode="json"),
            namespace=("charts",),
        )
    )
    return artifact


def _tool_result(document: PlotDocument, summary: str) -> str:
    values = ", ".join(
        f"{quarter}={value:g}k"
        for quarter, value in zip(QUARTERS, _values(document), strict=True)
    )
    return f"{summary} Current values: {values}."


@tool
async def show_quarterly_revenue(
    runtime: ToolRuntime[PersistentPlotAgentContext],
) -> str:
    """Read the stored quarterly revenue values and display the current chart."""
    store, context = _runtime_dependencies(runtime)
    document = await _load_document(store, context)
    artifact = _publish(document, context, runtime)
    return _tool_result(document, artifact.summary)


@tool
async def update_quarterly_revenue(
    updates: Annotated[list[RevenueUpdate], Field(min_length=1)],
    runtime: ToolRuntime[PersistentPlotAgentContext],
) -> str:
    """Apply one or more quarterly revenue edits and display the updated chart."""
    store, context = _runtime_dependencies(runtime)
    document = await _load_document(store, context)
    patch = {update.quarter.lower(): update.revenue for update in updates}
    updated = PlotDocument.model_validate({**document.model_dump(), **patch})

    if updated != document:
        await store.aput(
            _thread_namespace(context),
            ARTIFACT_KEY,
            updated.model_dump(mode="json"),
        )

    artifact = _publish(updated, context, runtime)
    return _tool_result(updated, artifact.summary)


def _chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        streaming=True,
        model_kwargs={"parallel_tool_calls": False},
    )


def create_persistent_plot_agent(
    store: BaseStore,
    model: BaseChatModel | None = None,
) -> PersistentPlotAgent:
    """Build the agent with its lifespan-managed Store."""
    return create_agent(
        model=model or _chat_model(),
        tools=[show_quarterly_revenue, update_quarterly_revenue],
        system_prompt=SYSTEM_PROMPT,
        context_schema=PersistentPlotAgentContext,
        store=store,
    )


def context_factory(
    request: ChatCompletionRequest,
    client_settings: ClientSettings | None,
) -> PersistentPlotAgentContext:
    user_id, session_id = _persistence_scope(request)
    plot_settings = (
        client_settings
        if isinstance(client_settings, PersistentPlotAgentSettings)
        else PersistentPlotAgentSettings()
    )
    return PersistentPlotAgentContext(
        user_id=user_id,
        session_id=session_id,
        settings=plot_settings,
    )


def _persistence_scope(request: ChatCompletionRequest) -> tuple[str, str]:
    if not request.user:
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message="user is required for persistent plot agent storage.",
                type="invalid_request_error",
                param="user",
                code="missing_persistence_scope",
            ),
        )
    session_id = (request.metadata or {}).get("session_id")
    if not session_id:
        raise OpenAIHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorObject(
                message="metadata.session_id is required for persistent plot agent storage.",
                type="invalid_request_error",
                param="metadata.session_id",
                code="missing_persistence_scope",
            ),
        )
    return request.user, session_id


def create_persistent_plot_agent_config(
    graph_factory: Callable[[], PersistentPlotAgent],
) -> GraphConfig:
    """Create the OpenAI-facing configuration for the agent."""
    return GraphConfig(
        graph=graph_factory,
        description="Uses an agent to inspect and edit a persistent revenue chart.",
        context_factory=context_factory,
        streamable_node_names=["model"],
        features={GraphFeature.CLIENT_EVENTS},
        client_settings=PersistentPlotAgentSettings,
    )


__all__ = [
    "ARTIFACT_KEY",
    "ChartArtifact",
    "PersistentPlotAgentContext",
    "PersistentPlotAgentSettings",
    "PlotDocument",
    "RevenueUpdate",
    "create_persistent_plot_agent",
    "create_persistent_plot_agent_config",
]
