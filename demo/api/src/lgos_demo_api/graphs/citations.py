"""Deterministic graph combining portable Markdown with OpenAI URL citations."""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.content import create_citation, create_text_block
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.utils.fake_llm import stream_fake_chat_response
from pydantic import BaseModel

SOURCE_TITLE = "LangGraph streaming documentation"
SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/streaming#custom-data"
IMAGE_TITLE = "MDN grapefruit image example"
IMAGE_URL = (
    "https://interactive-examples.mdn.mozilla.net/media/cc0-images/"
    "grapefruit-slice-332-332.jpg"
)
AUDIO_TITLE = "MDN audio example"
AUDIO_URL = (
    "https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3"
)
ANSWER = f"""This response showcases portable resource presentation:

- Read the [{SOURCE_TITLE}]({SOURCE_URL}).
- View the [{IMAGE_TITLE}]({IMAGE_URL}) inline:

  ![A grapefruit slice]({IMAGE_URL})

- Keep audio portable as an [{AUDIO_TITLE}]({AUDIO_URL})."""

CITATIONS = (
    (SOURCE_TITLE, SOURCE_URL),
    (IMAGE_TITLE, IMAGE_URL),
    (AUDIO_TITLE, AUDIO_URL),
)


class CitationState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def answer_with_citation(
    state: CitationState,
) -> dict[str, list[AIMessage]]:
    """Stream Markdown content and return its citations on the final message."""
    answer = await stream_fake_chat_response(
        ANSWER,
        prompt=str(state.messages[-1].content or ""),
    )
    citations = []
    for title, url in CITATIONS:
        start = answer.index(title)
        citations.append(
            create_citation(
                url=url,
                title=title,
                start_index=start,
                end_index=start + len(title) - 1,
                cited_text=title,
            )
        )
    return {
        "messages": [
            AIMessage(
                content_blocks=[create_text_block(text=answer, annotations=citations)]
            )
        ]
    }


workflow = StateGraph(CitationState)
workflow.add_node("answer_with_citation", answer_with_citation)
workflow.add_edge("answer_with_citation", END)
workflow.set_entry_point("answer_with_citation")

citation_graph = workflow.compile()

citation_graph_config = GraphConfig(
    graph=citation_graph,
    description=(
        "Demonstrates structured OpenAI URL citations with portable Markdown content."
    ),
    streamable_node_names=["answer_with_citation"],
)

__all__ = ["citation_graph", "citation_graph_config"]
