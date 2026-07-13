import pytest
from demo.api.graphs.citations import (
    ANSWER,
    AUDIO_URL,
    CITATIONS,
    IMAGE_URL,
    SOURCE_URL,
    citation_graph,
)
from langgraph.types import CustomStreamPart
from openai.types.chat.chat_completion_message import Annotation

from langgraph_openai_serve import GraphConfig, GraphRegistry, citation_slice
from langgraph_openai_serve.graph.runner import (
    LangGraphInterrupt,
    run_langgraph_stream,
)


@pytest.mark.anyio
async def test_citation_demo_streams_markdown_and_citations(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "citation-events": GraphConfig(
                graph=citation_graph,
                streamable_node_names=["answer_with_citation"],
            )
        }
    )
    request = make_request("citation-events", content="Show citations")

    events = [
        event
        async for event in run_langgraph_stream(
            "citation-events",
            request.messages,
            registry,
            request,
        )
    ]

    custom_events: list[CustomStreamPart] = [
        event for event in events if not isinstance(event, (str, LangGraphInterrupt))
    ]
    assert "".join(event for event in events if isinstance(event, str)) == ANSWER

    annotations = [Annotation.model_validate(event["data"]) for event in custom_events]
    actual_citations = [
        (
            ANSWER[citation_slice(annotation, ANSWER)],
            annotation.url_citation.title,
            annotation.url_citation.url,
        )
        for annotation in annotations
    ]
    expected_citations = [(title, title, url) for title, url in CITATIONS]
    assert actual_citations == expected_citations

    assert f"[LangGraph streaming documentation]({SOURCE_URL})" in ANSWER
    assert f"![A grapefruit slice]({IMAGE_URL})" in ANSWER
    assert f"[MDN audio example]({AUDIO_URL})" in ANSWER
