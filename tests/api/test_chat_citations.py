import pytest
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.types import CustomStreamPart
from openai import OpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
    citation_event,
)
from langgraph_openai_serve.api.chat.utils.events import annotation_from_custom_event
from tests.graph.support.schemas import MessageState

ANSWER = "Cited answer [1]"
SOURCE_START_INDEX = 13
SOURCE_END_INDEX = 16
SOURCE_TITLE = "Example source"
SOURCE_URL = "https://example.com/source"
ANNOTATION = {
    "type": "url_citation",
    "url_citation": {
        "start_index": SOURCE_START_INDEX,
        "end_index": SOURCE_END_INDEX,
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
                start_index=SOURCE_START_INDEX,
                end_index=SOURCE_END_INDEX,
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


def test_non_streaming_completion_includes_annotations(
    openai_client: OpenAI,
) -> None:
    response = openai_client.chat.completions.create(
        model="citations",
        messages=[{"role": "user", "content": "Cite this"}],
    )

    message = response.choices[0].message
    assert message.content == ANSWER
    assert message.annotations is not None
    assert [annotation.model_dump() for annotation in message.annotations] == [
        ANNOTATION
    ]
    assert ANSWER[SOURCE_START_INDEX:SOURCE_END_INDEX] == "[1]"


def test_streaming_completion_emits_annotations_on_final_delta(
    openai_client: OpenAI,
) -> None:
    chunks = list(
        openai_client.chat.completions.create(
            model="citations",
            messages=[{"role": "user", "content": "Cite this"}],
            stream=True,
        )
    )

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


def test_citation_must_refer_to_final_assistant_text() -> None:
    event = CustomStreamPart(
        type="custom",
        ns=(),
        data=citation_event(
            url=SOURCE_URL,
            title=SOURCE_TITLE,
            start_index=0,
            end_index=len(ANSWER) + 1,
        ),
    )

    with pytest.raises(ValueError, match="final assistant text"):
        annotation_from_custom_event(event, ANSWER)
