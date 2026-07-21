import pytest
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph

from lgos_demo_api.graphs.custom_io import custom_io_graph_config


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("user", "expected_user"),
    [
        pytest.param("demo-user", "demo-user", id="request-user"),
        pytest.param(None, "anonymous", id="anonymous"),
    ],
)
async def test_adapts_request_input_context_and_output(
    make_request,
    user: str | None,
    expected_user: str,
) -> None:
    request = make_request(
        "custom-input-output-context",
        content="Show me custom schemas.",
        user=user,
    )
    registry = GraphRegistry(
        registry={"custom-input-output-context": custom_io_graph_config}
    )

    result = await run_langgraph(
        request.model,
        request.messages,
        registry,
        request,
    )

    assert result.output == f"{expected_user} asked: Show me custom schemas."
