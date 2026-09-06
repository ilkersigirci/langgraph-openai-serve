from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.runner import run_langgraph

from lgos_demo_api.graphs.advanced_mcp import advanced_mcp_graph


async def test_async_factory_loads_and_calls_the_mock_mcp_tool(
    make_graph_input,
) -> None:
    graph_request, messages = make_graph_input(
        "advanced-mcp-tools",
        content="What is the weather in Istanbul?",
    )
    registry = GraphRegistry(
        registry={
            "advanced-mcp-tools": GraphConfig(
                graph=advanced_mcp_graph,
                description="DUMMY",
            )
        }
    )

    result = await run_langgraph(graph_request, messages, registry)

    assert result.text == (
        "The async mock MCP tool was loaded and called. "
        "It reported sunny weather in Istanbul."
    )
