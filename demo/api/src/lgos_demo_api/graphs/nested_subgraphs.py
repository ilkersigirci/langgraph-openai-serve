"""
Deterministic graph that streams, pauses, and resumes inside nested subgraphs.

The root graph runs a child subgraph, which streams a draft and then runs a
grandchild subgraph that pauses for review and streams the answer. One request
therefore exercises nested token streaming, nested status events, an interrupt
two levels deep, and its resumption, in the foreground or background, without a
model call.
"""

from collections.abc import Callable
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_openai_serve import GraphConfig, GraphFeature, status_event
from pydantic import BaseModel

from lgos_demo_api.utils.fake_llm import stream_fake_chat_response

DRAFT = "Draft from the child subgraph: {request}"
ANSWER = "The grandchild subgraph resumed with: {decision}"


class NestedSubgraphsState(BaseModel):
    """Every level shares this schema, so subgraphs run directly as nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    decision: str = ""


async def draft(state: NestedSubgraphsState) -> dict[str, list[AIMessage]]:
    get_stream_writer()(status_event("Drafting in the child subgraph"))
    request = state.messages[-1].text
    text = await stream_fake_chat_response(DRAFT.format(request=request), request)
    return {"messages": [AIMessage(content=text)]}


def review(_state: NestedSubgraphsState) -> dict[str, str]:
    # LangGraph reruns an interrupted node on resume, so this node only asks;
    # the draft and the answer stream in the nodes around it.
    decision = interrupt(
        {
            "question": "Approve the child subgraph's draft?",
            "choices": ["approve", "reject"],
            "allow_other": False,
        }
    )
    return {"decision": decision}


async def answer(state: NestedSubgraphsState) -> dict[str, list[AIMessage]]:
    response = ANSWER.format(decision=state.decision)
    text = await stream_fake_chat_response(response, response)
    return {"messages": [AIMessage(content=text)]}


def _sequence(
    *nodes: tuple[str, Callable[..., object] | CompiledStateGraph],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    # Subgraphs compile without a checkpointer and inherit the root's, which
    # stores their paused state under nested checkpoint namespaces.
    return (
        StateGraph(NestedSubgraphsState)
        .add_sequence(nodes)
        .add_edge(START, nodes[0][0])
        .add_edge(nodes[-1][0], END)
        .compile(checkpointer=checkpointer)
    )


def create_nested_subgraphs_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    grandchild = _sequence(("review", review), ("answer", answer))
    child = _sequence(("draft", draft), ("grandchild", grandchild))
    return _sequence(("child", child), checkpointer=checkpointer)


def create_nested_subgraphs_graph_config(graph: CompiledStateGraph) -> GraphConfig:
    """Create the nested subgraph demo config around its lifespan-managed graph."""
    return GraphConfig(
        graph=graph,
        description=(
            "Demonstrates streaming, status, human review, and background runs "
            "inside nested subgraphs without a model call."
        ),
        features={GraphFeature.BACKGROUND, GraphFeature.INTERRUPTS},
    )
