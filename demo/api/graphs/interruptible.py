from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt


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


interruptible_graph = (
    StateGraph(ApprovalState)
    .add_node("request_approval", request_approval)
    .set_entry_point("request_approval")
    .set_finish_point("request_approval")
    .compile(checkpointer=InMemorySaver())
)
