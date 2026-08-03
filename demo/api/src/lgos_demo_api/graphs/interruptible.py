from collections.abc import Callable
from typing import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_openai_serve import GraphConfig, GraphFeature
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest


class ApprovalState(TypedDict, total=False):
    request: str
    response: str


def request_approval(state: ApprovalState) -> dict[str, str]:
    decision = interrupt(
        {
            "question": "Approve this agent action?",
            "request": state["request"],
            "choices": ["approve", "reject"],
        }
    )

    if str(decision).strip().lower() in {"approve", "approved", "yes", "y"}:
        response = f"Approved agent action: {state['request']}"
    else:
        response = f"Rejected agent action: {state['request']}"

    return {"response": response}


def create_interruptible_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    return (
        StateGraph(ApprovalState)
        .add_node("request_approval", request_approval)
        .set_entry_point("request_approval")
        .set_finish_point("request_approval")
        .compile(checkpointer=checkpointer)
    )


def request_to_input(
    _request: ChatCompletionRequest,
    messages: list[BaseMessage],
) -> ApprovalState:
    return {"request": str(messages[-1].content or "")}


def output_to_text(output: ApprovalState) -> str:
    return output["response"]


def create_interruptible_graph_config(
    graph_factory: Callable[[], CompiledStateGraph],
) -> GraphConfig:
    """Create the interrupt demo config around its lifespan-managed graph."""
    return GraphConfig(
        graph=graph_factory,
        description=("Requests human approval through a checkpointed interrupt flow."),
        request_to_input=request_to_input,
        output_to_text=output_to_text,
        features={GraphFeature.INTERRUPTS},
    )


__all__ = ["create_interruptible_graph", "create_interruptible_graph_config"]
