"""Connect the demo gateway to the reusable Chainlit MCP bridge."""

import chainlit as cl
from chainlit.config import StreamableHttpMcpServer
from chainlit.mcp import McpConnection
from chainlit_utils.mcp import McpTools
from mcp import ClientSession

from lgos_chainlit.auth import gateway_credential
from lgos_chainlit.gateway import GatewayConfig

MCP_GATEWAY_NAME = "lgos-gateway"
mcp_tools = McpTools(MCP_GATEWAY_NAME, session_key="_lgos_mcp_gateway")


def mcp_gateway_config(gateway: GatewayConfig) -> StreamableHttpMcpServer:
    """
    Build Chainlit's connection to the gateway's aggregate MCP route.

    Each request sends the same credential as Responses: the shared key, or the
    signed-in user's delegated token.
    """
    return mcp_tools.server(f"{gateway.root_url}/mcp", api_key=gateway_credential)


@cl.on_mcp_connect
async def on_mcp_connect(connection: McpConnection, session: ClientSession) -> None:
    await mcp_tools.connect(connection, session)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession) -> None:
    await mcp_tools.disconnect(name, session)
