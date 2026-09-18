"""Connect the demo gateway to the reusable Chainlit MCP bridge."""

import chainlit as cl
from chainlit.config import StreamableHttpMcpServer
from chainlit.mcp import McpConnection
from chainlit_utils.mcp import McpTools
from mcp import ClientSession

from lgos_chainlit.gateway import GatewayConfig

MCP_GATEWAY_NAME = "lgos-gateway"
mcp_tools = McpTools(MCP_GATEWAY_NAME, session_key="_lgos_mcp_gateway")


def mcp_gateway_config(
    gateway: GatewayConfig,
    api_key: str,
) -> StreamableHttpMcpServer:
    """Build Chainlit's trusted connection to the selected gateway."""
    return mcp_tools.server(
        gateway.mcp_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )


@cl.on_mcp_connect
async def on_mcp_connect(connection: McpConnection, session: ClientSession) -> None:
    await mcp_tools.connect(connection, session)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession) -> None:
    await mcp_tools.disconnect(name, session)
