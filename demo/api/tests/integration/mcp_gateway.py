"""Shared live MCP gateway assertions."""

import json

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

EXPECTED_TOOL_NAMES = (
    "lgos_postgres-count_chainlit_users",
    "lgos_postgres-list_chainlit_conversation_counts",
    "lgos_postgres-summarize_chainlit_activity",
    "lgos_postgres-list_chainlit_activity_by_profile",
    "lgos_postgres-summarize_lgos_interrupted_runs",
    "lgos_postgres-list_lgos_interrupted_runs",
)


def _tool_row(result: CallToolResult) -> dict[str, object]:
    assert result.is_error is not True
    assert result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload = json.loads(block.text)
    rows = payload["data"]["statements"][0]["rows"]
    assert payload["success"] is True
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert isinstance(row, dict)
    return row


async def assert_postgres_mcp_contract(
    gateway_root: str,
    api_key: str,
    *,
    endpoint: str,
) -> None:
    """Smoke-test authentication, discovery, and both reporting domains."""
    url = f"{gateway_root.rstrip('/')}/{endpoint.lstrip('/')}"
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "lgos-demo-tests", "version": "1"},
        },
    }
    async with httpx2.AsyncClient(timeout=20.0) as anonymous:
        response = await anonymous.post(
            url,
            json=initialize,
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code == 401

    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx2.Timeout(20.0, read=30.0),
    ) as http:
        transport = streamable_http_client(url, http_client=http)
        async with Client(transport) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == set(EXPECTED_TOOL_NAMES)
            results = {
                name: await client.call_tool(name, {}) for name in EXPECTED_TOOL_NAMES
            }
            assert all(result.is_error is not True for result in results.values())

            users = _tool_row(results["lgos_postgres-count_chainlit_users"])
            user_count = users["user_count"]
            assert isinstance(user_count, int | str)
            assert int(user_count) >= 0

            interrupted = _tool_row(
                results["lgos_postgres-summarize_lgos_interrupted_runs"]
            )
            interrupt_count = interrupted["total_interrupted_runs"]
            assert isinstance(interrupt_count, int | str)
            assert int(interrupt_count) >= 0
