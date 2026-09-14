from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph

from lgos_demo_api.graphs.mcp_mock import mcp_mock_graph


async def test_async_factory_loads_and_calls_the_mock_mcp_tool(
    make_graph_input,
) -> None:
    graph_request, messages = make_graph_input(
        "mcp-mock",
        content="What is the weather in Istanbul?",
    )
    registry = GraphRegistry(
        registry={
            "mcp-mock": GraphConfig(
                graph=mcp_mock_graph,
                description="DUMMY",
            )
        }
    )

    result = await run_langgraph(graph_request, messages, registry)

    assert result.text == (
        "The async mock MCP tool was loaded and called. "
        "It reported sunny weather in Istanbul."
    )
