"""Small, dependency-free example of loading MCP-style tools."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph_openai_serve import GraphConfig


class MockToolCallingChatModel(FakeMessagesListChatModel):
    """Fake chat model that supports tool binding for the deterministic demo."""

    def bind_tools(
        self, tools: list[BaseTool], **kwargs: Any
    ) -> "MockToolCallingChatModel":  # ty: ignore[invalid-method-override]
        return self


class MockMCPClient:
    """Minimal stand-in for an MCP client that discovers tools asynchronously."""

    async def get_tools(self) -> list[BaseTool]:
        return [mock_weather_tool]


@tool
async def mock_weather_tool(city: str) -> str:
    """Get deterministic mock weather for a city."""
    return f"The mock MCP weather service says it is sunny in {city}."


async def mcp_mock_graph() -> CompiledStateGraph:
    """Build an agent after asynchronously loading one mock MCP tool."""
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


mcp_mock_graph_config = GraphConfig(
    graph=mcp_mock_graph,
    description=(
        "Demonstrates async MCP-style tool discovery with no network or credentials."
    ),
)

__all__ = ["mcp_mock_graph", "mcp_mock_graph_config"]
