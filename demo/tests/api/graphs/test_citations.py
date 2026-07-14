import pytest
from demo.api.graphs.citations import citation_graph
from openai.types.chat.chat_completion_message import Annotation

from langgraph_openai_serve import GraphConfig, GraphRegistry, citation_slice
from langgraph_openai_serve.graph.runner import run_langgraph_stream

EXPECTED_CITATIONS = [
    (
        "LangGraph streaming documentation",
        "https://docs.langchain.com/oss/python/langgraph/streaming#custom-data",
    ),
    (
        "MDN grapefruit image example",
        "https://interactive-examples.mdn.mozilla.net/media/cc0-images/"
        "grapefruit-slice-332-332.jpg",
    ),
    (
        "MDN audio example",
        "https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3",
    ),
]


@pytest.mark.anyio
async def test_streams_portable_markdown_with_anchored_citations(make_request) -> None:
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
            request.model,
            request.messages,
            registry,
            request,
        )
    ]

    answer = "".join(event for event in events if isinstance(event, str))
    annotations = [
        Annotation.model_validate(event["data"])
        for event in events
        if not isinstance(event, str)
    ]

    assert [
        (
            answer[citation_slice(annotation, answer)],
            annotation.url_citation.title,
            annotation.url_citation.url,
        )
        for annotation in annotations
    ] == [(title, title, url) for title, url in EXPECTED_CITATIONS]
    assert "[LangGraph streaming documentation](" in answer
    assert "![A grapefruit slice](" in answer
    assert "[MDN audio example](" in answer
