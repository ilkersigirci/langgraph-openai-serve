"""State and runtime context shared by the advanced graph and its subgraphs."""

from dataclasses import dataclass
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve import GraphRequest
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

Intent = Literal["chat", "research", "save", "research_and_save"]


class IntentDecision(BaseModel):
    """Private structured output used to route one user turn."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent = Field(
        description="The single workflow that best matches the latest user request."
    )


@dataclass(frozen=True, slots=True)
class AdvancedContext:
    request: GraphRequest


class Note(TypedDict):
    id: str
    filename: str
    content: str
    vector_store_id: str


class NoteReceipt(TypedDict):
    filename: str
    vector_store_id: str
    content_sha256: str
    file_id: str | None
    status: Literal["upload_pending", "uploaded", "indexed", "index_failed"]


class AdvancedState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Intent
    note: Note | None
    feedback: str | None
    decision: Literal["approve", "reject", "revise"] | None
    receipt: NoteReceipt | None
    research_used: bool
    web_search_used: bool
    terminal: bool


def terminal_message(message: AIMessage) -> AIMessage | None:
    """Preserve provider refusal/incomplete outcomes from private model steps."""
    if message.response_metadata.get("status") == "incomplete":
        return message.model_copy(
            update={"content": [], "tool_calls": [], "invalid_tool_calls": []}
        )
    refusals = [
        block
        for block in message.content_blocks
        if block["type"] == "non_standard"
        and isinstance(block.get("value"), dict)
        and block["value"].get("type") == "refusal"
    ]
    if not refusals and not message.additional_kwargs.get("refusal"):
        return None
    return message.model_copy(
        update={
            "content": refusals,
            "tool_calls": [],
            "response_metadata": {
                **message.response_metadata,
                "output_version": "v1",
            },
        }
    )


# ty does not recognize TypedDict class attributes in LangGraph's StateLike bound.
AdvancedGraph = CompiledStateGraph[
    AdvancedState, AdvancedContext, AdvancedState, AdvancedState  # ty: ignore[invalid-type-arguments]
]


__all__ = [
    "AdvancedContext",
    "AdvancedGraph",
    "AdvancedState",
    "Intent",
    "IntentDecision",
    "Note",
    "NoteReceipt",
    "terminal_message",
]
