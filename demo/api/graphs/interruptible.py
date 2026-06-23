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
            "question": "Approve this request?",
            "request": state["request"],
        }
    )

    if str(decision).strip().lower() in {"approve", "approved", "yes", "y"}:
        response = f"Approved: {state['request']}"
    else:
        response = f"Not approved: {state['request']}. Resume value: {decision}"

    return {"response": response}


interruptible_graph = (
    StateGraph(ApprovalState)
    .add_node("request_approval", request_approval)
    .set_entry_point("request_approval")
    .set_finish_point("request_approval")
    .compile(checkpointer=InMemorySaver())
)
