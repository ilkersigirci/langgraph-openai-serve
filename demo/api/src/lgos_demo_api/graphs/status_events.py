"""Deterministic graph showcasing portable status events."""

import asyncio
from typing import Annotated, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig, status_event
from pydantic import BaseModel

ANSWER = "The media workflow completed successfully."
STATUS_EVENT_DELAY_SECONDS = 0.5


class StatusEventState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def prepare_media(
    state: StatusEventState,
) -> dict[str, list[AIMessage]]:
    """Simulate a long-running workflow with user-facing status."""
    writer = get_stream_writer()

    writer(status_event("Generating audio", namespace=("media",)))
    await asyncio.sleep(STATUS_EVENT_DELAY_SECONDS)

    writer(status_event("Calculating embeddings", namespace=("media",)))
    await asyncio.sleep(STATUS_EVENT_DELAY_SECONDS)

    writer(status_event("Media ready", done=True, namespace=("media",)))
    model = GenericFakeChatModel(messages=iter([ANSWER]))
    answer = await model.ainvoke(state.messages)
    return {"messages": [answer]}


workflow = StateGraph(StatusEventState)
workflow.add_node("prepare_media", prepare_media)
workflow.add_edge("prepare_media", END)
workflow.set_entry_point("prepare_media")

status_event_graph = workflow.compile()
status_event_graph_config = GraphConfig(
    graph=status_event_graph,
    streamable_node_names=["prepare_media"],
)

__all__ = ["status_event_graph_config"]
