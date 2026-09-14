"""Synchronize the demo's native Open WebUI MCP connections."""

from typing import Any

import httpx

from .functions.generic.gateway import MCP_GATEWAY_ID, GatewayConfig

PUBLIC_READ_GRANT = {
    "principal_type": "user",
    "principal_id": "*",
    "permission": "read",
}
MCP_GATEWAY_TOOL_ID = f"server:mcp:{MCP_GATEWAY_ID}"


def mcp_gateway_connection(
    gateway: GatewayConfig,
    api_key: str,
) -> dict[str, Any]:
    """Build Open WebUI's native connection to the selected gateway."""
    return {
        "url": gateway.mcp_url,
        "path": "",
        "type": "mcp",
        "auth_type": "bearer",
        "headers": None,
        "key": api_key,
        "config": {
            "enable": True,
            # The gateway credential owns the tool grant.
            "function_name_filter_list": [],
            "access_grants": [PUBLIC_READ_GRANT],
        },
        "info": {
            "id": MCP_GATEWAY_ID,
            "name": "LGOS Gateway",
            "description": "MCP tools authorized by the configured gateway key.",
        },
    }


def sync_mcp_gateway(
    client: httpx.Client,
    *,
    gateway: GatewayConfig,
    api_key: str,
) -> str:
    """Upsert the managed MCP gateway while preserving unrelated connections."""
    payload = client.get("/api/v1/configs/tool_servers").raise_for_status().json()
    connections = (
        payload.get("TOOL_SERVER_CONNECTIONS") if isinstance(payload, dict) else None
    )
    if not isinstance(connections, list):
        msg = "Open WebUI tool-server configuration returned invalid data."
        raise TypeError(msg)

    desired = mcp_gateway_connection(gateway, api_key)
    managed = [
        connection
        for connection in connections
        if _connection_id(connection) == MCP_GATEWAY_ID
    ]
    if managed == [desired]:
        return "unchanged"

    updated = [
        connection
        for connection in connections
        if _connection_id(connection) != MCP_GATEWAY_ID
    ]
    updated.append(desired)
    client.post(
        "/api/v1/configs/tool_servers",
        json={"TOOL_SERVER_CONNECTIONS": updated},
    ).raise_for_status()
    return "updated" if managed else "created"


def _connection_id(connection: object) -> str | None:
    if not isinstance(connection, dict) or not isinstance(
        info := connection.get("info"), dict
    ):
        return None
    connection_id = info.get("id")
    return connection_id if isinstance(connection_id, str) else None


__all__ = [
    "MCP_GATEWAY_ID",
    "MCP_GATEWAY_TOOL_ID",
    "PUBLIC_READ_GRANT",
    "mcp_gateway_connection",
    "sync_mcp_gateway",
]
