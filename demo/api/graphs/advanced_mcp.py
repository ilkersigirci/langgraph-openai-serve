"""Demo graph for async MCP-style tool loading."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool


class MockToolCallingChatModel(FakeMessagesListChatModel):
    """Fake chat model that supports LangGraph tool binding for the demo."""

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any):
        return self


class MockMCPClient:
    """Small stand-in for MultiServerMCPClient used by the demo."""

    async def get_tools(self) -> list[BaseTool]:
        return [mock_weather_tool]


@tool
async def mock_weather_tool(city: str) -> str:
    """Get mock weather data for a city."""
    return f"The mock MCP weather service says it is sunny in {city}."


async def advanced_mcp_graph():
    """Build a ReAct graph after asynchronously loading MCP-style tools."""
    tools = await MockMCPClient().get_tools()
    model = MockToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "mock_weather_tool",
                        "args": {"city": "Istanbul"},
                        "id": "mock-call-1",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "The async mock MCP tool was loaded and called. "
                    "It reported sunny weather in Istanbul."
                )
            ),
        ]
    )
    return create_agent(model=model, tools=tools)
