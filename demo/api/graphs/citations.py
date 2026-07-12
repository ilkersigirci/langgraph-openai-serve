"""Deterministic graph that emits an OpenAI-compatible URL citation."""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from langgraph_openai_serve import citation_event
from langgraph_openai_serve.utils.fake_llm import stream_fake_chat_response

ANSWER = (
    "LangGraph custom stream mode lets nodes emit user-defined data while a graph "
    "runs. [1]"
)
SOURCE_TITLE = "LangGraph streaming documentation"
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/streaming#custom-data"


class CitationState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def answer_with_citation(
    state: CitationState,
) -> dict[str, list[AIMessage]]:
    """Emit citation metadata and a deterministic streamed answer."""
    marker_start = ANSWER.index("[1]")
    get_stream_writer()(
        citation_event(
            url=SOURCE_URL,
            title=SOURCE_TITLE,
            start_index=marker_start,
            end_index=marker_start + len("[1]"),
        )
    )
    answer = await stream_fake_chat_response(
        ANSWER,
        prompt=str(state.messages[-1].content or ""),
    )
    return {"messages": [AIMessage(content=answer)]}


workflow = StateGraph(CitationState)
workflow.add_node("answer_with_citation", answer_with_citation)
workflow.add_edge("answer_with_citation", END)
workflow.set_entry_point("answer_with_citation")

citation_graph = workflow.compile()

__all__ = ["citation_graph"]
