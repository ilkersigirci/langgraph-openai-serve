import pytest
from langchain_core.messages import AIMessage
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

from lgos_demo_api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from lgos_demo_api.graphs.subgraphs.keyword import create_keyword_graph
from lgos_demo_api.graphs.subgraphs.schemas import KeywordState

API_ANSWER = (
    "API contract: OpenAI request messages were adapted into native graph input; "
    "native graph output is rendered back as assistant text; "
    "streamable nested node names can be exposed safely"
)
DOCS_ANSWER = (
    "Docs specialist: covered subgraph, routing; nested keyword subgraph selected "
    "`subgraph`, `routing`; keyword subgraph shared the docs state channels"
)


def _registry() -> GraphRegistry:
    return GraphRegistry(
        registry={"complex-subgraphs": create_complex_subgraphs_graph_config()}
    )


async def test_keyword_extraction_falls_back_to_general() -> None:
    graph = create_keyword_graph()
    graph_view = graph.get_graph()

    assert "prepare_keyword_context" in graph_view.nodes
    assert (
        "extract_keywords",
        "prepare_keyword_context",
    ) in {(edge.source, edge.target) for edge in graph_view.edges}

    result = await graph.ainvoke(KeywordState(normalized_question="Hello."))

    assert result["keywords"] == ["general"]
    assert result["checks"] == ["nested keyword subgraph selected `general`"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param(
            "Show OpenAI adapter streaming with nested subgraphs.",
            API_ANSWER,
            id="api",
        ),
        pytest.param(
            "Show nested subgraph routing docs.",
            DOCS_ANSWER,
            id="docs",
        ),
    ],
)
async def test_routes_to_the_expected_specialist(
    make_request,
    question: str,
    expected: str,
) -> None:
    request = make_request("complex-subgraphs", content=question)

    graph_request, messages, _ = decode_responses_request(request)

    result = await run_langgraph(graph_request, messages, _registry())

    assert isinstance(result.output, AIMessage)
    assert result.output.text == expected


async def test_streaming_matches_non_streaming_for_nested_output(
    make_request,
) -> None:
    request = make_request(
        "complex-subgraphs",
        content="Show nested subgraph routing docs.",
    )

    graph_request, messages, _ = decode_responses_request(request)

    events = [
        event
        async for event in run_langgraph_stream(graph_request, messages, _registry())
    ]

    assert [
        event["data"]["event"]["data"]["description"]
        for event in events
        if isinstance(event, dict)
    ] == ["Selected keywords: subgraph, routing"]

    streamed = "".join(event for event in events if isinstance(event, str))
    complete = await run_langgraph(graph_request, messages, _registry())

    assert isinstance(complete.output, AIMessage)
    assert streamed == complete.output.text == DOCS_ANSWER
