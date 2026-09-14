from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph_openai_serve import (
    ClientFunctionTool,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    GraphRequest,
)
from langgraph_openai_serve.graph.runner import run_langgraph

from lgos_demo_api.graphs import mcp_postgres as graph_module
from lgos_demo_api.graphs import simple_external_tools

COUNT_USERS = ClientFunctionTool(
    name="lgos_postgres-count_chainlit_users",
    description="Count all Chainlit users.",
    parameters={"type": "object", "properties": {}},
    strict=True,
)
UNRELATED_TOOL = ClientFunctionTool(
    name="delete_file",
    description="Delete a local file.",
    parameters={"type": "object"},
    strict=True,
)


class RecordingModel:
    """Small model double that records tool binding."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bound_tools: Sequence[dict[str, Any]] | None = None
        self.bound_tool_choice: object = None

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any]],
        *,
        tool_choice: object = None,
        **_: object,
    ) -> "RecordingModel":
        self.bound_tools = tools
        self.bound_tool_choice = tool_choice
        return self

    async def ainvoke(self, _: Sequence[BaseMessage]) -> AIMessage:
        return self.response


def _request(*tools: ClientFunctionTool) -> GraphRequest:
    return GraphRequest(
        model="mcp-postgres",
        metadata={},
        user=None,
        tools=tools,
        tool_choice=None,
        parallel_tool_calls=False,
    )


def _registry() -> GraphRegistry:
    return GraphRegistry(
        registry={
            "mcp-postgres": GraphConfig(
                graph=graph_module.mcp_postgres_graph,
                description="DUMMY",
                request_to_input=graph_module.request_to_input,
            )
        }
    )


def test_graph_advertises_gateway_mcp_tools() -> None:
    assert graph_module.mcp_postgres_graph_config.features == {GraphFeature.MCP_TOOLS}


async def test_database_turn_requires_an_allowlisted_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel(
        AIMessage(
            content="",
            tool_calls=[{"name": COUNT_USERS.name, "args": {}, "id": "call-1"}],
        )
    )
    monkeypatch.setattr(simple_external_tools, "ChatOpenAI", lambda **_: model)

    result = await run_langgraph(
        _request(COUNT_USERS, UNRELATED_TOOL),
        [HumanMessage(content="How many Chainlit users do I have?")],
        _registry(),
    )

    assert model.bound_tool_choice == "required"
    assert [tool["function"]["name"] for tool in model.bound_tools or []] == [
        COUNT_USERS.name,
    ]
    assert isinstance(result, AIMessage)
    assert result.tool_calls[0]["name"] == COUNT_USERS.name


async def test_database_result_allows_a_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel(AIMessage(content="There are 12 Chainlit users."))
    monkeypatch.setattr(simple_external_tools, "ChatOpenAI", lambda **_: model)
    messages = [
        HumanMessage(content="How many Chainlit users do I have?"),
        AIMessage(
            content="",
            tool_calls=[{"name": COUNT_USERS.name, "args": {}, "id": "call-1"}],
        ),
        ToolMessage(content='[{"user_count":12}]', tool_call_id="call-1"),
    ]

    result = await run_langgraph(_request(COUNT_USERS), messages, _registry())

    assert model.bound_tool_choice == "auto"
    assert result.content == "There are 12 Chainlit users."
