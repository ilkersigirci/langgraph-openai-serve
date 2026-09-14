"""Native Chainlit MCP server discovery and execution."""

import json
from dataclasses import dataclass

import chainlit as cl
from chainlit.config import StreamableHttpMcpServer
from chainlit.mcp import McpConnection
from mcp import ClientSession
from mcp.types import Tool
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall

from lgos_chainlit.gateway import GatewayConfig

MCP_GATEWAY_NAME = "lgos-gateway"
_MCP_GATEWAY_SESSION_ATTR = "_lgos_mcp_gateway"


@dataclass(frozen=True)
class _ConnectedGateway:
    session: ClientSession
    tools: tuple[Tool, ...]


def mcp_gateway_config(
    gateway: GatewayConfig,
    api_key: str,
) -> StreamableHttpMcpServer:
    """Build Chainlit's trusted connection to the selected gateway."""
    return StreamableHttpMcpServer(
        name=MCP_GATEWAY_NAME,
        type="streamable-http",
        url=gateway.mcp_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def _connected_gateway() -> _ConnectedGateway | None:
    gateway = cl.user_session.get(_MCP_GATEWAY_SESSION_ATTR)
    return gateway if isinstance(gateway, _ConnectedGateway) else None


@cl.on_mcp_connect
async def on_mcp_connect(connection: McpConnection, session: ClientSession) -> None:
    """Discover tools authorized by the configured gateway credential."""
    if connection.name != MCP_GATEWAY_NAME:
        return
    discovered = await session.list_tools()
    cl.user_session.set(
        _MCP_GATEWAY_SESSION_ATTR,
        _ConnectedGateway(session=session, tools=tuple(discovered.tools)),
    )


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession) -> None:
    """Forget tools belonging to the disconnected gateway session."""
    gateway = _connected_gateway()
    if name != MCP_GATEWAY_NAME or gateway is None or gateway.session is not session:
        return
    cl.user_session.set(_MCP_GATEWAY_SESSION_ATTR, None)


def mcp_response_tools() -> list[FunctionToolParam]:
    """Return schemas discovered through the configured gateway."""
    gateway = _connected_gateway()
    if gateway is None:
        return []

    response_tools = []
    for tool in gateway.tools:
        response_tool: FunctionToolParam = {
            "type": "function",
            "name": tool.name,
            "parameters": tool.inputSchema,
            "strict": False,
        }
        if tool.description is not None:
            response_tool["description"] = tool.description
        response_tools.append(response_tool)
    return response_tools


async def execute_mcp_tool(
    call: ResponseFunctionToolCall,
) -> dict[str, str]:
    """Execute one graph-requested tool through Chainlit's owned MCP session."""
    gateway = _connected_gateway()
    if gateway is None or not any(tool.name == call.name for tool in gateway.tools):
        msg = f"MCP tool is not connected: {call.name}"
        raise ValueError(msg)
    try:
        arguments = json.loads(call.arguments)
    except ValueError as exc:
        msg = f"MCP tool call contains invalid arguments: {call.name}"
        raise ValueError(msg) from exc
    if not isinstance(arguments, dict):
        msg = f"MCP tool arguments must be an object: {call.name}"
        raise ValueError(msg)

    async with cl.Step(name=call.name, type="tool") as step:
        step.input = arguments
        result = await gateway.session.call_tool(call.name, arguments)
        output = result.model_dump_json(by_alias=True, exclude_none=True)
        step.output = output
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": output,
    }


__all__ = [
    "MCP_GATEWAY_NAME",
    "execute_mcp_tool",
    "mcp_gateway_config",
    "mcp_response_tools",
    "on_mcp_connect",
    "on_mcp_disconnect",
]
