"""Deterministic graph combining output from multiple streaming nodes."""

from operator import add
from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.utils.fake_llm import stream_fake_chat_response
from pydantic import BaseModel, Field

FIRST_CONTRIBUTION = "The first node contributed this sentence. "
SECOND_CONTRIBUTION = "The second node contributed this sentence."
ANSWER = FIRST_CONTRIBUTION + SECOND_CONTRIBUTION


class MultiNodeStreamingState(BaseModel):
    """Collect completed contributions in graph execution order."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer_parts: Annotated[list[str], add] = Field(default_factory=list)


async def _contribute(
    state: MultiNodeStreamingState,
    response: str,
) -> dict[str, list[str]]:
    contribution = await stream_fake_chat_response(
        response,
        prompt=str(state.messages[-1].content or ""),
    )
    return {"answer_parts": [contribution]}


async def write_first_contribution(
    state: MultiNodeStreamingState,
) -> dict[str, list[str]]:
    """Stream and retain the first part of the final answer."""
    return await _contribute(state, FIRST_CONTRIBUTION)


async def write_second_contribution(
    state: MultiNodeStreamingState,
) -> dict[str, list[str]]:
    """Stream and retain the second part of the final answer."""
    return await _contribute(state, SECOND_CONTRIBUTION)


def assemble_answer(
    state: MultiNodeStreamingState,
) -> dict[str, list[AIMessage]]:
    """Combine the ordered contributions into the final assistant turn."""
    return {"messages": [AIMessage(content="".join(state.answer_parts))]}


multi_node_streaming_graph = (
    StateGraph(MultiNodeStreamingState)
    .add_node("write_first_contribution", write_first_contribution)
    .add_node("write_second_contribution", write_second_contribution)
    .add_node("assemble_answer", assemble_answer)
    .add_edge(START, "write_first_contribution")
    .add_edge("write_first_contribution", "write_second_contribution")
    .add_edge("write_second_contribution", "assemble_answer")
    .add_edge("assemble_answer", END)
    .compile()
)

multi_node_streaming_graph_config = GraphConfig(
    graph=multi_node_streaming_graph,
    description=(
        "Combines streamed output from multiple nodes into one assistant message."
    ),
    streamable_node_names=[
        "write_first_contribution",
        "write_second_contribution",
    ],
)

__all__ = ["multi_node_streaming_graph_config"]
