"""Deterministic graph demonstrating native Responses terminal outcomes."""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig
from pydantic import BaseModel

HELP = 'Send "refusal" or "incomplete" to select a native Responses outcome.'
REFUSAL = "I cannot help with bypassing safety controls."
PARTIAL_ANSWER = "This answer stopped before it could finish."


class ResponseOutcomeState(BaseModel):
    """Messages used to select a deterministic response outcome."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def respond_with_outcome(
    state: ResponseOutcomeState,
) -> dict[str, list[AIMessage]]:
    """Return completion, refusal, or incomplete metadata for the final message."""
    outcome = state.messages[-1].text.strip().lower()
    if outcome == "refusal":
        return {
            "messages": [
                AIMessage(
                    content_blocks=[
                        {
                            "type": "non_standard",
                            "value": {"type": "refusal", "refusal": REFUSAL},
                        }
                    ]
                )
            ]
        }
    if outcome == "incomplete":
        return {
            "messages": [
                AIMessage(
                    content=PARTIAL_ANSWER,
                    response_metadata={
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                )
            ]
        }
    return {"messages": [AIMessage(content=HELP)]}


workflow = StateGraph(ResponseOutcomeState)
workflow.add_node("respond_with_outcome", respond_with_outcome)
workflow.add_edge("respond_with_outcome", END)
workflow.set_entry_point("respond_with_outcome")

response_outcome_graph = workflow.compile()
response_outcome_graph_config = GraphConfig(
    graph=response_outcome_graph,
    description="Demonstrates native refusal and incomplete Responses outcomes.",
)

__all__ = ["response_outcome_graph", "response_outcome_graph_config"]
