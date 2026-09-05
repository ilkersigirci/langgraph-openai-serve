from langchain_core.messages import AIMessage
from langgraph_openai_serve import GraphConfig, GraphRegistry, citation_slice
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.graph.citations import citations_from_message
from langgraph_openai_serve.graph.runner import run_langgraph_stream

from lgos_demo_api.graphs.citations import citation_graph

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


async def test_streams_portable_markdown_with_anchored_citations(make_request) -> None:
    registry = GraphRegistry(
        registry={
            "citation-events": GraphConfig(
                graph=citation_graph,
                description="DUMMY",
                streamable_node_names=["answer_with_citation"],
            )
        }
    )
    request = make_request("citation-events", content="Show citations")

    graph_request, messages, _ = decode_responses_request(request)

    events = [
        event async for event in run_langgraph_stream(graph_request, messages, registry)
    ]

    answer = "".join(event for event in events if isinstance(event, str))
    final_message = events[-1]
    assert isinstance(final_message, AIMessage)
    annotations = citations_from_message(final_message)

    assert [
        (
            answer[
                citation_slice(
                    annotation["start_index"], annotation["end_index"], answer
                )
            ],
            annotation["title"],
            annotation["url"],
        )
        for annotation in annotations
    ] == [(title, title, url) for title, url in EXPECTED_CITATIONS]
    for index, (title, url) in enumerate(EXPECTED_CITATIONS, start=1):
        assert f"[{title}]({url}) [{index}]" in answer
    assert "![A grapefruit slice](" in answer
