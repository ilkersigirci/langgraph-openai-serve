import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent, Tool
from openai.types.responses import ResponseFunctionToolCall

from lgos_chainlit.gateway import gateway_config
from lgos_chainlit.utils import mcp as mcp_module


@pytest.mark.parametrize(
    ("gateway_type", "url"),
    [
        ("litellm", "https://gateway.example/mcp/"),
        ("bifrost", "https://gateway.example/mcp"),
    ],
)
def test_gateway_config_builds_each_native_mcp_endpoint(
    gateway_type: str,
    url: str,
) -> None:
    server = mcp_module.mcp_gateway_config(
        gateway_config(gateway_type, "https://gateway.example/"),
        "secret",
    )

    assert (server.name, server.url) == (mcp_module.MCP_GATEWAY_NAME, url)
    assert server.headers == {"Authorization": "Bearer secret"}


async def test_native_mcp_session_discovers_and_executes_read_only_tools(
    chainlit_context,
    monkeypatch,
) -> None:
    class StepDouble:
        def __init__(self, **_: object) -> None:
            self.input: object = None
            self.output: object = None

        async def __aenter__(self) -> "StepDouble":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    allowed = Tool(
        name="database_report",
        description="Run a database report.",
        inputSchema={"type": "object", "properties": {}},
    )
    session = AsyncMock()
    session.list_tools.return_value = SimpleNamespace(tools=[allowed])
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text='[{"user_count":12}]')]
    )
    monkeypatch.setattr(mcp_module.cl, "Step", StepDouble)

    server_name = mcp_module.MCP_GATEWAY_NAME
    connection = SimpleNamespace(name=server_name)
    await mcp_module.on_mcp_connect(connection, session)
    response_tools = mcp_module.mcp_response_tools()
    assert [tool["name"] for tool in response_tools] == [allowed.name]

    output = await mcp_module.execute_mcp_tool(
        ResponseFunctionToolCall(
            id="fc-1",
            call_id="call-1",
            name=allowed.name,
            arguments="{}",
            status="completed",
            type="function_call",
        ),
    )

    session.call_tool.assert_awaited_once_with(allowed.name, {})
    assert output["call_id"] == "call-1"
    assert json.loads(output["output"])["content"][0]["text"] == '[{"user_count":12}]'

    replacement = AsyncMock()
    replacement.list_tools.return_value = SimpleNamespace(tools=[allowed])
    await mcp_module.on_mcp_connect(connection, replacement)
    await mcp_module.on_mcp_disconnect(server_name, session)
    assert [tool["name"] for tool in mcp_module.mcp_response_tools()] == [allowed.name]

    await mcp_module.on_mcp_disconnect(server_name, replacement)
    assert mcp_module.mcp_response_tools() == []
