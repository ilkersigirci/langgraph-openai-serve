"""Durable human approval, with no model credentials or external side effects."""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_openai_serve import InvalidRequestError
from pydantic import BaseModel


class ApprovalState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    decision: Literal["approve", "reject"] | None = None


ApprovalGraph = CompiledStateGraph[ApprovalState, None, ApprovalState, ApprovalState]


def review(state: ApprovalState) -> dict[str, str]:
    # LangGraph restarts this node on resume. Keep side effects in later nodes.
    answer = interrupt(
        {
            "question": "Approve this request?",
            "request": state.messages[-1].text,
            "choices": ["approve", "reject"],
            "allow_other": False,
        }
    )
    if isinstance(answer, str) and answer.strip().lower() in {"approve", "reject"}:
        return {"decision": answer.strip().lower()}
    msg = "Approval response must be approve or reject."
    raise InvalidRequestError(msg, param="input", code="invalid_approval_decision")


def finish(state: ApprovalState) -> dict[str, list[AIMessage]]:
    content = (
        "Request approved." if state.decision == "approve" else "Request rejected."
    )
    return {"messages": [AIMessage(content=content)]}


def create_approval_graph(checkpointer: BaseCheckpointSaver) -> ApprovalGraph:
    workflow = StateGraph(ApprovalState)
    workflow.add_node("review", review)
    workflow.add_node("finish", finish)
    workflow.add_edge(START, "review")
    workflow.add_edge("review", "finish")
    workflow.add_edge("finish", END)
    return workflow.compile(checkpointer=checkpointer)
