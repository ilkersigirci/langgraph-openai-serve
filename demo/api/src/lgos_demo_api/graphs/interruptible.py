from collections.abc import Callable
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRequest
from langgraph_openai_serve.graph.interrupt.coordination import RunCoordinator

ReviewOutcome = Literal["approve", "reject", "feedback"]


class ReviewState(TypedDict, total=False):
    request: str
    review_outcome: ReviewOutcome
    reviewer_feedback: str
    refund_executed: bool
    customer_notified: bool


def review_refund(state: ReviewState) -> dict[str, str]:
    response = interrupt(
        {
            "action": "refund",
            "question": "How should the refund be handled?",
            "request": state["request"],
            "choices": ["approve", "reject"],
            "allow_other": True,
        }
    )
    if not isinstance(response, str) or not (normalized := response.strip()):
        msg = "Refund review response must be a non-empty string."
        raise ValueError(msg)
    decision = normalized.lower()
    if decision in {"approve", "reject"}:
        return {"review_outcome": decision}
    return {
        "review_outcome": "feedback",
        "reviewer_feedback": normalized,
    }


def route_after_refund(
    state: ReviewState,
) -> Literal["execute_refund", "__end__"]:
    if state["review_outcome"] == "approve":
        return "execute_refund"
    return "__end__"


def execute_refund(state: ReviewState) -> dict[str, bool]:
    return {"refund_executed": state["review_outcome"] == "approve"}


def notify_customer(state: ReviewState) -> dict[str, bool]:
    return {"customer_notified": state.get("refund_executed", False)}


def create_interruptible_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    graph = StateGraph(ReviewState)  # ty: ignore[invalid-argument-type]
    graph.add_node("review_refund", review_refund)
    graph.add_node("execute_refund", execute_refund)
    graph.add_node("notify_customer", notify_customer)
    graph.add_edge(START, "review_refund")
    graph.add_conditional_edges("review_refund", route_after_refund)
    graph.add_edge("execute_refund", "notify_customer")
    graph.add_edge("notify_customer", END)
    return graph.compile(checkpointer=checkpointer)


def request_to_input(
    _request: GraphRequest,
    messages: list[BaseMessage],
) -> ReviewState:
    return {"request": str(messages[-1].content or "")}


def output_to_message(output: ReviewState) -> AIMessage:
    refund_executed = output.get("refund_executed", False)
    customer_notified = output.get("customer_notified", False)
    executed_actions = []
    if refund_executed:
        executed_actions.append("Refund")
    if customer_notified:
        executed_actions.append("Customer notification")
    lines = [
        f"Review workflow for: {output['request']}",
        f"- Refund: {output['review_outcome']}",
        f"- Customer notification: {'sent' if customer_notified else 'skipped'}",
        f"- Executed actions: {', '.join(executed_actions) or 'none'}",
    ]
    if feedback := output.get("reviewer_feedback"):
        lines.append(f"- Reviewer feedback: {feedback}")
    return AIMessage(content="\n".join(lines))


def create_interruptible_graph_config(
    graph_factory: Callable[[], CompiledStateGraph],
    run_coordinator: RunCoordinator,
) -> GraphConfig:
    """Create the interrupt demo config around its lifespan-managed graph."""
    return GraphConfig(
        graph=graph_factory,
        description=(
            "Demonstrates durable choice-or-text human review before protected "
            "actions execute."
        ),
        request_to_input=request_to_input,
        output_to_message=output_to_message,
        features={GraphFeature.INTERRUPTS},
        run_coordinator=run_coordinator,
    )


__all__ = [
    "ReviewState",
    "create_interruptible_graph",
    "create_interruptible_graph_config",
    "output_to_message",
]
