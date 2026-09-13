from importlib.metadata import version

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph_openai_serve import GraphRequest

from lgos_demo_api.graphs import server_tool
from lgos_demo_api.graphs.server_tool import lgos_package_version


async def test_package_version_reads_the_server_environment() -> None:
    result = await lgos_package_version.ainvoke(" OpenAI ")

    assert result == [
        {
            "type": "custom_tool_call_output",
            "output": f"openai=={version('openai')}",
        }
    ]


async def test_unsupported_package_is_actionable() -> None:
    result = await lgos_package_version.ainvoke("requests")

    assert result == [
        {
            "type": "custom_tool_call_output",
            "output": (
                "Unsupported package: requests. Choose one of: "
                "langgraph-openai-serve, langgraph, langchain, "
                "langchain-openai, openai."
            ),
        }
    ]


async def test_graph_does_not_execute_an_unselected_tool(
    monkeypatch: pytest.MonkeyPatch, make_tool_calling_model
) -> None:
    async def unexpected_search(*args):
        pytest.fail("The unselected search tool must not execute.")

    model = make_tool_calling_model(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "latest OpenAI Python SDK release"},
                    "id": "call_unselected",
                }
            ],
        ),
        AIMessage(content="Search is not enabled."),
    )
    monkeypatch.setattr(server_tool, "ChatOpenAI", lambda **kwargs: model)
    monkeypatch.setattr(server_tool, "search_web", unexpected_search)
    graph = server_tool.create_server_tool_graph()
    request = GraphRequest(
        model="server-tool",
        metadata={},
        user=None,
        tools=(),
        server_tools=("lgos_package_version",),
        tool_choice="auto",
        parallel_tool_calls=None,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Is this OpenAI SDK current?")]},
        context=server_tool.context_factory(request, None),
    )

    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert tool_result.tool_call_id == "call_unselected"
    assert "web_search" in tool_result.text
    assert result["messages"][-1].text == "Search is not enabled."
