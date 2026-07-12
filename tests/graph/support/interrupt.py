from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from tests.graph.support.schemas import MessageState

DEFAULT_INTERRUPT_PAYLOAD = {"question": "Approve?"}


def make_interrupt_graph(
    payload: dict[str, Any] | None = None,
    *,
    checkpointer: BaseCheckpointSaver,
    response_prefix: str = "resumed",
) -> Any:
    interrupt_payload = DEFAULT_INTERRUPT_PAYLOAD if payload is None else payload

    def ask(state: MessageState):
        answer = interrupt(interrupt_payload)
        return {"messages": [AIMessage(content=f"{response_prefix}:{answer}")]}

    return (
        StateGraph(MessageState)
        .add_node("ask", ask)
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile(checkpointer=checkpointer)
    )
