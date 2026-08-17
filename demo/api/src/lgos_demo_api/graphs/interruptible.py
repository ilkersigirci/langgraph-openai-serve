import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_openai_serve import GraphConfig, GraphFeature
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.graph.interrupt.coordination import RunCoordinator

ApprovalDecision = Literal["approve", "reject"]


def merge_shared_request(current: str, update: str) -> str:
    """Merge the identical request returned by parallel nested subgraphs."""
    if not current:
        return update
    if current != update:
        raise ValueError("Parallel approval steps received different requests.")
    return current


class ApprovalResult(TypedDict):
    action: str
    decision: ApprovalDecision


class ApprovalState(TypedDict, total=False):
    request: Annotated[str, merge_shared_request]
    approvals: Annotated[list[ApprovalResult], operator.add]


@dataclass(frozen=True)
class ApprovalSpec:
    action: str
    question: str


APPROVAL_SPECS = (
    ApprovalSpec(action="Refund", question="Approve the refund?"),
    ApprovalSpec(
        action="Customer notification",
        question="Approve notifying the customer?",
    ),
)


def create_approval_subgraph(spec: ApprovalSpec) -> CompiledStateGraph:
    """Create one reusable nested approval step."""

    def request_approval(state: ApprovalState) -> dict[str, list[ApprovalResult]]:
        decision = interrupt(
            {
                "question": spec.question,
                "request": state["request"],
                "choices": ["approve", "reject"],
            }
        )
        normalized_decision = str(decision).strip().lower()
        if normalized_decision not in {"approve", "reject"}:
            raise ValueError("Approval decision must be 'approve' or 'reject'.")

        return {
            "approvals": [
                ApprovalResult(
                    action=spec.action,
                    decision=normalized_decision,
                )
            ]
        }

    return (
        StateGraph(ApprovalState)  # ty: ignore[invalid-argument-type]
        .add_node("request_approval", request_approval)
        .add_edge(START, "request_approval")
        .add_edge("request_approval", END)
        .compile()
    )


def create_interruptible_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    graph = StateGraph(ApprovalState)  # ty: ignore[invalid-argument-type]
    for index, spec in enumerate(APPROVAL_SPECS):
        node_name = f"approval_{index}"
        graph.add_node(node_name, create_approval_subgraph(spec))
        graph.add_edge(START, node_name)
        graph.add_edge(node_name, END)
    return graph.compile(checkpointer=checkpointer)


def request_to_input(
    _request: ChatCompletionRequest,
    messages: list[BaseMessage],
) -> ApprovalState:
    return {"request": str(messages[-1].content or "")}


def output_to_text(output: ApprovalState) -> str:
    decisions = {result["action"]: result["decision"] for result in output["approvals"]}
    lines = [f"Approval results for: {output['request']}"]
    lines.extend(
        f"- {spec.action}: {decisions[spec.action]}" for spec in APPROVAL_SPECS
    )
    return "\n".join(lines)


def create_interruptible_graph_config(
    graph_factory: Callable[[], CompiledStateGraph],
    run_coordinator: RunCoordinator,
) -> GraphConfig:
    """Create the interrupt demo config around its lifespan-managed graph."""
    return GraphConfig(
        graph=graph_factory,
        description=(
            "Requests one atomic approval batch from parallel nested graph steps."
        ),
        request_to_input=request_to_input,
        output_to_text=output_to_text,
        features={GraphFeature.INTERRUPTS},
        run_coordinator=run_coordinator,
    )


__all__ = [
    "APPROVAL_SPECS",
    "ApprovalState",
    "create_approval_subgraph",
    "create_interruptible_graph",
    "create_interruptible_graph_config",
    "output_to_text",
]
