import pytest
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.types import CustomStreamPart
from openai import AsyncOpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
    citation_event,
    citation_slice,
)
from langgraph_openai_serve.api.chat.utils.events import annotation_from_custom_event
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
        get_stream_writer()(
            citation_event(
                url=SOURCE_URL,
                title=SOURCE_TITLE,
                span=SOURCE_SPAN,
            )
        )
        return {"messages": [await model.ainvoke(state["messages"])]}

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
                streamable_node_names=["generate"],
            )
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


@pytest.fixture
def fastapi_app() -> FastAPI:
    return citation_app()


@pytest.mark.anyio
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


@pytest.mark.parametrize(
    "span",
    [
        pytest.param((-1, 1), id="negative-start"),
        pytest.param((0, 0), id="empty"),
        pytest.param((2, 1), id="reversed"),
    ],
)
def test_citation_event_rejects_invalid_half_open_spans(
    span: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="valid non-empty"):
        citation_event(url=SOURCE_URL, title=SOURCE_TITLE, span=span)


@pytest.mark.anyio
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
    event = CustomStreamPart(
        type="custom",
        ns=(),
        data=citation_event(
            url=SOURCE_URL,
            title=SOURCE_TITLE,
            span=(0, len(ANSWER) + 1),
        ),
    )

    with pytest.raises(ValueError, match="final assistant text"):
        annotation_from_custom_event(event, ANSWER)
