import pytest
from demo.api.graphs.complex_subgraphs import create_complex_subgraphs_graph_config
from demo.api.graphs.subgraphs.keyword import extract_keywords
from demo.api.graphs.subgraphs.schemas import KeywordState

from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph, run_langgraph_stream

API_ANSWER = (
    "API contract: OpenAI chat messages were adapted into native graph input; "
    "native graph output is rendered back as assistant text; "
    "streamable nested node names can be exposed safely"
)
DOCS_ANSWER = (
    "Docs specialist: covered subgraph, routing; nested keyword subgraph selected "
    "`subgraph`, `routing`; keyword subgraph shared the docs state channels"
)
DOCS_STREAM = f"Keyword subgraph: selected subgraph, routing\n{DOCS_ANSWER}\n"


def _registry() -> GraphRegistry:
    return GraphRegistry(
        registry={"complex-subgraphs": create_complex_subgraphs_graph_config()}
    )


@pytest.mark.anyio
async def test_keyword_extraction_falls_back_to_general() -> None:
    result = await extract_keywords(KeywordState(normalized_question="Hello."))

    assert result == {
        "keywords": ["general"],
        "checks": ["nested keyword subgraph selected `general`"],
    }


@pytest.mark.anyio
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

    result = await run_langgraph(
        request.model,
        request.messages,
        _registry(),
        request,
    )

    assert result.output == expected


@pytest.mark.anyio
async def test_streams_the_selected_nested_subgraphs_in_order(make_request) -> None:
    request = make_request(
        "complex-subgraphs",
        content="Show nested subgraph routing docs.",
    )

    events = [
        event
        async for event in run_langgraph_stream(
            request.model,
            request.messages,
            _registry(),
            request,
        )
    ]

    assert "".join(event for event in events if isinstance(event, str)) == DOCS_STREAM
