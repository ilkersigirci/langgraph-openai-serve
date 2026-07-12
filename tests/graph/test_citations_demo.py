import pytest
from demo.api.graphs.citations import ANSWER, SOURCE_TITLE, SOURCE_URL, citation_graph
from langgraph.types import CustomStreamPart
from openai.types.chat.chat_completion_message import Annotation

from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import (
    LangGraphInterrupt,
    run_langgraph,
    run_langgraph_stream,
)


@pytest.mark.anyio
async def test_citation_demo_streams_text_and_citation(make_request) -> None:
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
    assert len(custom_events) == 1
    citation = Annotation.model_validate(custom_events[0]["data"]).url_citation
    assert citation.title == SOURCE_TITLE
    assert citation.url == SOURCE_URL
    assert ANSWER[citation.start_index : citation.end_index] == "[1]"


@pytest.mark.anyio
async def test_citation_demo_preserves_messages_during_non_streaming_run(
    make_request,
) -> None:
    registry = GraphRegistry(
        registry={"citation-events": GraphConfig(graph=citation_graph)}
    )
    request = make_request("citation-events", content="Show citations")

    invocation = await run_langgraph(
        "citation-events",
        request.messages,
        registry,
        request,
    )

    assert invocation.output == ANSWER
    assert len(invocation.custom_events) == 1
