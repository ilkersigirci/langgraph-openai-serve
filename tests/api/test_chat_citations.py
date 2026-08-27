import pytest
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.content import create_citation, create_text_block
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from openai import AsyncOpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
    citation_slice,
)
from langgraph_openai_serve.api.chat.utils.responses import annotations_from_message
from tests.graph.support.schemas import MessageState

ANSWER = "Cited answer with source"
CITATION_TEXT = "source"
SOURCE_START = ANSWER.index(CITATION_TEXT)
SOURCE_SPAN = (SOURCE_START, SOURCE_START + len(CITATION_TEXT))
SOURCE_TITLE = "Example source"
SOURCE_URL = "https://example.com/source"
ANNOTATION = {
    "type": "url_citation",
    "url_citation": {
        "start_index": SOURCE_SPAN[0],
        "end_index": SOURCE_SPAN[1] - 1,
        "title": SOURCE_TITLE,
        "url": SOURCE_URL,
    },
}


def citation_app() -> FastAPI:
    model = FakeListChatModel(responses=[ANSWER])

    async def generate(state: MessageState):
        get_stream_writer()({"type": "progress", "data": {"percent": 50}})
        response = await model.ainvoke(state["messages"])
        return {
            "messages": [
                AIMessage(
                    content_blocks=[
                        create_text_block(
                            text=response.text,
                            annotations=[
                                create_citation(
                                    url=SOURCE_URL,
                                    title=SOURCE_TITLE,
                                    start_index=SOURCE_SPAN[0],
                                    end_index=SOURCE_SPAN[1] - 1,
                                    cited_text=CITATION_TEXT,
                                )
                            ],
                        )
                    ]
                )
            ]
        }

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    registry = GraphRegistry(
        registry={
            "citations": GraphConfig(
                graph=graph,
                description="DUMMY",
                streamable_node_names=["generate"],
            )
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


@pytest.fixture
def fastapi_app() -> FastAPI:
    return citation_app()


async def test_non_streaming_completion_uses_openai_inclusive_end_index(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.chat.completions.create(
        model="citations",
        messages=[{"role": "user", "content": "Cite this"}],
    )

    message = response.choices[0].message
    assert message.content == ANSWER
    assert message.annotations is not None
    assert [annotation.model_dump() for annotation in message.annotations] == [
        ANNOTATION
    ]
    assert ANSWER[citation_slice(message.annotations[0], ANSWER)] == CITATION_TEXT


async def test_streaming_completion_emits_annotations_on_final_delta(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await openai_client.chat.completions.create(
        model="citations",
        messages=[{"role": "user", "content": "Cite this"}],
        stream=True,
        metadata={"langgraph_stream_events": "v1"},
    )
    chunks = [chunk async for chunk in stream]

    annotated_chunks = [
        chunk
        for chunk in chunks
        if (chunk.choices[0].delta.model_extra or {}).get("annotations")
    ]
    annotation_deltas = [
        (chunk.choices[0].delta.model_extra or {})["annotations"]
        for chunk in annotated_chunks
    ]
    assert annotation_deltas == [[ANNOTATION]]
    assert annotated_chunks[0].choices[0].finish_reason == "stop"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == ANSWER
    assert all(
        "langgraph_openai_serve" not in (chunk.model_extra or {}) for chunk in chunks
    )


def test_citation_must_refer_to_final_assistant_text() -> None:
    message = AIMessage(
        content_blocks=[
            create_text_block(
                text=ANSWER,
                annotations=[
                    create_citation(
                        url=SOURCE_URL,
                        title=SOURCE_TITLE,
                        start_index=0,
                        end_index=len(ANSWER),
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValueError, match="final assistant text"):
        annotations_from_message(message)


def test_citation_indices_are_offset_across_text_blocks() -> None:
    prefix = "First block. "
    cited_text = "Second"
    message = AIMessage(
        content_blocks=[
            create_text_block(text=prefix),
            create_text_block(
                text=f"{cited_text} block.",
                annotations=[
                    create_citation(
                        url=SOURCE_URL,
                        title=SOURCE_TITLE,
                        start_index=0,
                        end_index=len(cited_text) - 1,
                        cited_text=cited_text,
                    )
                ],
            ),
        ]
    )

    annotation = annotations_from_message(message)[0]

    assert annotation.url_citation.start_index == len(prefix)
    assert message.text[citation_slice(annotation, message.text)] == cited_text


def test_citation_indices_must_match_cited_text() -> None:
    message = AIMessage(
        content_blocks=[
            create_text_block(
                text=ANSWER,
                annotations=[
                    create_citation(
                        url=SOURCE_URL,
                        title=SOURCE_TITLE,
                        start_index=0,
                        end_index=4,
                        cited_text=CITATION_TEXT,
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValueError, match="match cited_text"):
        annotations_from_message(message)
