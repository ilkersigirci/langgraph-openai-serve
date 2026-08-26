"""A store-backed chart that survives otherwise stateless graph runs."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha256
from typing import Annotated, Literal

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphFeature,
    client_event,
)
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from pydantic import BaseModel, ConfigDict, Field, JsonValue

ARTIFACT_KEY = "quarterly-revenue"
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
REVENUE_UPDATE_PATTERN = re.compile(
    r"\b(?:set|change|update)\s+(q[1-4])\s+(?:to|=)\s+\$?(\d+(?:\.\d+)?)\s*k?\b",
    re.IGNORECASE,
)


class PlotDocument(BaseModel):
    """Canonical editable chart data stored independently of either UI."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1] = 1
    q1: float = Field(default=120, ge=0)
    q2: float = Field(default=180, ge=0)
    q3: float = Field(default=150, ge=0)
    q4: float = Field(default=230, ge=0)


class PersistentPlotSettings(ClientSettings):
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


class PlotlyArtifact(BaseModel):
    """Versioned chart payload shared by the graph and UI adapters."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    kind: Literal["plotly"] = "plotly"
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    figure: dict[str, JsonValue]


class PersistentPlotState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@dataclass(frozen=True, slots=True)
class PersistentPlotContext:
    user_id: str | None
    session_id: str | None
    settings: PersistentPlotSettings = dataclass_field(
        default_factory=PersistentPlotSettings
    )


PersistentPlotGraph = CompiledStateGraph[
    PersistentPlotState,
    PersistentPlotContext,
    PersistentPlotState,
    PersistentPlotState,
]


def _currency_symbol(currency: Literal["USD", "EUR"]) -> str:
    return "$" if currency == "USD" else "€"


def _artifact(
    document: PlotDocument,
    settings: PersistentPlotSettings,
) -> PlotlyArtifact:
    values = [document.q1, document.q2, document.q3, document.q4]
    highest_index = values.index(max(values))
    title = "Quarterly revenue"
    symbol = _currency_symbol(settings.currency)
    summary = f"{QUARTERS[highest_index]} is highest at {symbol}{max(values):g}k."
    trace: dict[str, JsonValue] = {
        "type": "bar" if settings.chart_type == "bar" else "scatter",
        "name": "Revenue",
        "x": list(QUARTERS),
        "y": values,
        "marker": {"color": "#6366f1"},
        "showlegend": settings.show_legend,
    }
    if settings.chart_type == "line":
        trace["mode"] = "lines+markers"
        trace["line"] = {"color": "#6366f1"}
    return PlotlyArtifact(
        id=ARTIFACT_KEY,
        title=title,
        summary=summary,
        figure={
            "data": [trace],
            "layout": {
                "title": {"text": title},
                "xaxis": {"title": {"text": "Quarter"}},
                "yaxis": {
                    "title": {"text": f"Revenue ({settings.currency}, thousands)"}
                },
                "showlegend": settings.show_legend,
            },
        },
    )


def _thread_namespace(context: PersistentPlotContext) -> tuple[str, ...] | None:
    if context.user_id is None or context.session_id is None:
        return None
    scope = sha256(f"{context.user_id}\0{context.session_id}".encode()).hexdigest()
    return ("demo", "persistent-plot", "threads", scope)


def _prompt(state: PersistentPlotState) -> str:
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content.lower()
    return ""


def _revenue_update(prompt: str) -> tuple[str, float] | None:
    match = REVENUE_UPDATE_PATTERN.search(prompt)
    if match is None:
        return None
    return match.group(1).lower(), float(match.group(2))


async def show_plot(
    state: PersistentPlotState,
    runtime: Runtime[PersistentPlotContext],
) -> dict[str, list[AIMessage]]:
    """Load, edit, and publish one durable chart document."""
    store = runtime.store
    if store is None:
        msg = "The persistent plot graph requires a LangGraph store."
        raise RuntimeError(msg)

    prompt = _prompt(state)
    thread_namespace = _thread_namespace(runtime.context)
    item = (
        await store.aget(thread_namespace, ARTIFACT_KEY)
        if thread_namespace is not None
        else None
    )
    document = PlotDocument.model_validate(item.value) if item is not None else None
    document = document or PlotDocument()
    changed = item is None
    update_message: str | None = None

    update = _revenue_update(prompt)
    if update is not None:
        quarter, revenue = update
        symbol = _currency_symbol(runtime.context.settings.currency)
        if getattr(document, quarter) == revenue:
            update_message = f"{quarter.upper()} is already {symbol}{revenue:g}k."
        else:
            document = PlotDocument.model_validate(
                {
                    **document.model_dump(),
                    quarter: revenue,
                }
            )
            changed = True
            update_message = f"I set {quarter.upper()} to {symbol}{revenue:g}k."

    if thread_namespace is not None and changed:
        await store.aput(
            thread_namespace,
            ARTIFACT_KEY,
            document.model_dump(mode="json"),
        )

    artifact = _artifact(document, runtime.context.settings)
    get_stream_writer()(
        client_event(
            "artifact",
            artifact.model_dump(mode="json"),
            namespace=("plots",),
        )
    )

    answer = " ".join(part for part in (update_message, artifact.summary) if part)
    model = GenericFakeChatModel(messages=iter([answer]))
    chunks: list[str] = []
    async for chunk in model.astream(state.messages):
        chunks.append(str(chunk.content))
    return {"messages": [AIMessage(content="".join(chunks))]}


def create_persistent_plot_graph(store: BaseStore) -> PersistentPlotGraph:
    """Compile the demo graph with its lifespan-managed store."""
    return (
        StateGraph(PersistentPlotState, context_schema=PersistentPlotContext)
        .add_node("show_plot", show_plot)
        .set_entry_point("show_plot")
        .add_edge("show_plot", END)
        .compile(store=store)
    )


def context_factory(
    request: ChatCompletionRequest,
    _client_settings: ClientSettings | None,
) -> PersistentPlotContext:
    session_id = (request.metadata or {}).get("session_id")
    settings = (
        _client_settings
        if isinstance(_client_settings, PersistentPlotSettings)
        else PersistentPlotSettings()
    )
    return PersistentPlotContext(
        user_id=request.user,
        session_id=session_id or None,
        settings=settings,
    )


def output_to_text(output: object) -> str:
    """Render the final graph state for a non-streaming completion."""
    state = PersistentPlotState.model_validate(output)
    return str(state.messages[-1].content) if state.messages else ""


def create_persistent_plot_graph_config(
    graph_factory: Callable[[], PersistentPlotGraph],
) -> GraphConfig:
    """Create the config around the lifespan-managed graph."""
    return GraphConfig(
        graph=graph_factory,
        description="Edits a thread-scoped Plotly chart across stateless requests.",
        context_factory=context_factory,
        output_to_text=output_to_text,
        streamable_node_names=["show_plot"],
        features={GraphFeature.CLIENT_EVENTS},
        client_settings=PersistentPlotSettings,
    )


__all__ = [
    "ARTIFACT_KEY",
    "PersistentPlotContext",
    "PersistentPlotSettings",
    "PlotDocument",
    "PlotlyArtifact",
    "create_persistent_plot_graph",
    "create_persistent_plot_graph_config",
]
