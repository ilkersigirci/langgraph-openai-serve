"""State and runtime context shared by the advanced graph and its subgraph."""

from dataclasses import dataclass
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve import ClientSettings, GraphRequest
from pydantic import Field
from typing_extensions import TypedDict


class AdvancedSettings(ClientSettings):
    save_note: bool = Field(
        default=False,
        title="Save a research note",
        description="Review a Markdown note before adding it to the knowledge base.",
    )


@dataclass(frozen=True, slots=True)
class AdvancedContext:
    request: GraphRequest
    settings: AdvancedSettings


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
    note: Note
    feedback: str
    decision: Literal["approve", "reject", "revise"]
    receipt: NoteReceipt
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
    "AdvancedSettings",
    "AdvancedState",
    "Note",
    "NoteReceipt",
    "terminal_message",
]
