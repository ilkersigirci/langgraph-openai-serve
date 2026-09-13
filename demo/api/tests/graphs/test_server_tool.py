from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph_openai_serve import GraphRequest

from lgos_demo_api.graphs import server_tool
from lgos_demo_api.graphs.server_tool import lgos_current_time


async def test_current_time_uses_the_lgos_clock() -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    result = await lgos_current_time.ainvoke("Asia/Tokyo")
    after = datetime.now(timezone.utc)
    assert isinstance(result, list)
    assert result[0]["type"] == "custom_tool_call_output"
    output = result[0]["output"]
    assert isinstance(output, str)
    actual = datetime.fromisoformat(output.removeprefix("Asia/Tokyo: "))
    assert before <= actual <= after
    assert actual.utcoffset().total_seconds() == 9 * 3600


async def test_unknown_timezone_is_actionable() -> None:
    result = await lgos_current_time.ainvoke("not/a/timezone")
    assert result == [
        {
            "type": "custom_tool_call_output",
            "output": (
                "Unknown timezone: not/a/timezone. "
                "Use an IANA name such as Europe/Istanbul."
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
                    "args": {"query": "current time"},
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
        server_tools=("lgos_current_time",),
        tool_choice="auto",
        parallel_tool_calls=None,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What time is it?")]},
        context=server_tool.context_factory(request, None),
    )

    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert tool_result.tool_call_id == "call_unselected"
    assert "web_search" in tool_result.text
    assert result["messages"][-1].text == "Search is not enabled."
