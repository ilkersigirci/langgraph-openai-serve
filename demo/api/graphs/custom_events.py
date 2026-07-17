"""Deterministic graph showcasing public custom stream events."""

import asyncio
from typing import Annotated, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from langgraph_openai_serve import GraphConfig, client_event

ANSWER = (
    "OpenAI compatibility stays intact: assistant text uses standard delta.content, "
    "while opt-in client events use the namespaced chunk extension."
)
SHOWCASE_EVENT_DELAY_SECONDS = 0.25
VALIDATION_EVENT_CHUNK_INDEX = 4
WRITING_EVENT_CHUNK_INDEX = 12


class CustomEventShowcaseState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def build_compatibility_report(
    state: CustomEventShowcaseState,
) -> dict[str, list[AIMessage]]:
    """Stream an answer while publishing a small event timeline."""
    writer = get_stream_writer()
    writer(
        client_event(
            "status",
            {"message": "Planning compatibility report"},
            namespace=("research",),
        )
    )
    await asyncio.sleep(SHOWCASE_EVENT_DELAY_SECONDS)

    writer(
        client_event(
            "progress",
            {
                "stage": "research",
                "completed": 1,
                "total": 3,
                "message": "Reading the compatibility guide",
            },
            namespace=("research",),
        )
    )
    await asyncio.sleep(SHOWCASE_EVENT_DELAY_SECONDS)

    prompt = str(state.messages[-1].content or "")
    model = GenericFakeChatModel(messages=iter([ANSWER]))
    chunks: list[str] = []
    index = 0
    async for chunk in model.astream([HumanMessage(content=prompt)]):
        chunks.append(str(chunk.content))
        if index == VALIDATION_EVENT_CHUNK_INDEX:
            writer(
                client_event(
                    "progress",
                    {
                        "stage": "validation",
                        "completed": 2,
                        "total": 3,
                        "message": "Checking the streaming contract",
                    },
                    namespace=("research",),
                )
            )
            await asyncio.sleep(SHOWCASE_EVENT_DELAY_SECONDS)
        elif index == WRITING_EVENT_CHUNK_INDEX:
            writer(
                client_event(
                    "progress",
                    {
                        "stage": "writing",
                        "completed": 3,
                        "total": 3,
                        "message": "Finishing the report",
                    },
                    namespace=("research",),
                )
            )
            await asyncio.sleep(SHOWCASE_EVENT_DELAY_SECONDS)
        index += 1

    writer(
        client_event(
            "artifact",
            {
                "id": "compatibility-brief",
                "kind": "report",
                "title": "OpenAI compatibility brief",
            },
            namespace=("research", "report"),
        )
    )
    return {"messages": [AIMessage(content="".join(chunks))]}


workflow = StateGraph(CustomEventShowcaseState)
workflow.add_node("build_compatibility_report", build_compatibility_report)
workflow.add_edge("build_compatibility_report", END)
workflow.set_entry_point("build_compatibility_report")

custom_event_showcase_graph = workflow.compile()
custom_event_showcase_graph_config = GraphConfig(
    graph=custom_event_showcase_graph,
    streamable_node_names=["build_compatibility_report"],
)

__all__ = ["custom_event_showcase_graph_config"]
