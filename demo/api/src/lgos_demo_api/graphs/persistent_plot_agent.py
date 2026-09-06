"""A persistent chart managed by a LangChain agent."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Any, Literal

from fastapi import status
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphRequest,
    NamedFunctionToolChoice,
)
from langgraph_openai_serve.core.errors import OpenAIHTTPException
from openai import AsyncOpenAI
from openai.types.shared import ErrorObject
from plotly import graph_objects as go
from pydantic import BaseModel, ConfigDict, Field

from lgos_demo_api.settings import settings

ARTIFACT_KEY = "quarterly-revenue"
DISPLAY_FILE_TOOL_NAME = "display_file"
PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
Quarter = Literal["Q1", "Q2", "Q3", "Q4"]
SYSTEM_PROMPT = """You manage one persistent quarterly revenue chart.

For reads or display requests, use show_quarterly_revenue. For edits, call
update_quarterly_revenue directly; it reads the current values itself. Put every
edit requested in one update call, and never call both tools for the same
request. Never invent stored values. If the latest input is a display_file result,
do not call another tool; acknowledge the result concisely. After any other tool
call, answer concisely and mention the important result.
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


class DisplayFile(BaseModel):
    """Arguments for the client-owned file display function."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    media_type: Literal["application/vnd.plotly.v1+json"] = PLOTLY_MEDIA_TYPE
    title: str = Field(min_length=1)
    alt: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class PersistentPlotAgentContext:
    user_id: str
    session_id: str
    settings: PersistentPlotAgentSettings
    display_file_available: bool = False


PersistentPlotAgent = CompiledStateGraph[Any, PersistentPlotAgentContext, Any, Any]


def _currency_symbol(currency: Literal["USD", "EUR"]) -> str:
    return "$" if currency == "USD" else "€"


def _values(document: PlotDocument) -> list[float]:
    return [document.q1, document.q2, document.q3, document.q4]


def _summary(
    document: PlotDocument,
    chart_settings: PersistentPlotAgentSettings,
) -> str:
    values = _values(document)
    highest_index = values.index(max(values))
    symbol = _currency_symbol(chart_settings.currency)
    return f"{QUARTERS[highest_index]} is highest at {symbol}{max(values):g}k."


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


def _render_plotly(
    document: PlotDocument,
    chart_settings: PersistentPlotAgentSettings,
) -> bytes:
    """Serialize the interactive figure using Plotly's native JSON format."""
    values = _values(document)
    trace = (
        go.Scatter(x=QUARTERS, y=values, mode="lines+markers", name="Revenue")
        if chart_settings.chart_type == "line"
        else go.Bar(x=QUARTERS, y=values, name="Revenue")
    )
    figure = go.Figure(trace)
    figure.update_layout(
        title="Quarterly revenue",
        xaxis_title="Quarter",
        yaxis_title=f"Revenue ({chart_settings.currency}, thousands)",
        showlegend=chart_settings.show_legend,
        template="plotly_white",
    )
    return figure.to_json().encode()


async def _publish(
    document: PlotDocument,
    context: PersistentPlotAgentContext,
) -> DisplayFile | None:
    if not context.display_file_available:
        return None

    content = _render_plotly(document, context.settings)
    digest = sha256(content).hexdigest()[:12]
    filename = f"quarterly-revenue-{digest}.plotly.json"
    async with AsyncOpenAI(
        base_url=settings.FILES_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
    ) as files_client:
        uploaded = await files_client.files.create(
            file=(filename, content, PLOTLY_MEDIA_TYPE),
            purpose="user_data",
        )
    return DisplayFile(
        file_id=uploaded.id,
        filename=filename,
        title="Quarterly revenue",
        alt=_summary(document, context.settings),
    )


def _tool_result(document: PlotDocument, summary: str) -> str:
    values = ", ".join(
        f"{quarter}={value:g}k"
        for quarter, value in zip(QUARTERS, _values(document), strict=True)
    )
    return f"{summary} Current values: {values}."


@tool(response_format="content_and_artifact")
async def show_quarterly_revenue(
    runtime: ToolRuntime[PersistentPlotAgentContext],
) -> tuple[str, DisplayFile | None]:
    """Read the stored quarterly revenue values and display the current chart."""
    store, context = _runtime_dependencies(runtime)
    document = await _load_document(store, context)
    summary = _summary(document, context.settings)
    return _tool_result(document, summary), await _publish(document, context)


@tool(response_format="content_and_artifact")
async def update_quarterly_revenue(
    updates: Annotated[list[RevenueUpdate], Field(min_length=1)],
    runtime: ToolRuntime[PersistentPlotAgentContext],
) -> tuple[str, DisplayFile | None]:
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

    summary = _summary(updated, context.settings)
    return _tool_result(updated, summary), await _publish(updated, context)


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
    request: GraphRequest,
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
        display_file_available=_display_file_available(request),
    )


def _display_file_available(request: GraphRequest) -> bool:
    if request.tool_choice == "none":
        return False
    if isinstance(request.tool_choice, NamedFunctionToolChoice):
        return request.tool_choice.name == DISPLAY_FILE_TOOL_NAME
    return any(tool.name == DISPLAY_FILE_TOOL_NAME for tool in request.tools)


def output_to_message(output: Any) -> AIMessage:
    """Prefer the latest deterministic display request over agent prose."""
    raw_messages = (
        output.get("messages")
        if isinstance(output, Mapping)
        else getattr(output, "messages", None)
    )
    if not isinstance(raw_messages, Sequence) or not raw_messages:
        msg = "Persistent plot output must contain messages."
        raise TypeError(msg)

    for message in reversed(raw_messages):
        if not isinstance(message, ToolMessage) or not isinstance(
            message.artifact, DisplayFile
        ):
            continue
        call_id = sha256(message.artifact.file_id.encode()).hexdigest()[:24]
        return AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    name=DISPLAY_FILE_TOOL_NAME,
                    args=message.artifact.model_dump(mode="json"),
                    id=f"lg_display_{call_id}",
                )
            ],
        )

    final = raw_messages[-1]
    if not isinstance(final, AIMessage):
        msg = "Persistent plot output must end with an AIMessage."
        raise TypeError(msg)
    return final


def _persistence_scope(request: GraphRequest) -> tuple[str, str]:
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
    session_id = request.metadata.get("session_id")
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
        client_settings=PersistentPlotAgentSettings,
        output_to_message=output_to_message,
    )


__all__ = [
    "ARTIFACT_KEY",
    "DISPLAY_FILE_TOOL_NAME",
    "DisplayFile",
    "PersistentPlotAgentContext",
    "PersistentPlotAgentSettings",
    "PlotDocument",
    "RevenueUpdate",
    "create_persistent_plot_agent",
    "create_persistent_plot_agent_config",
    "output_to_message",
]
