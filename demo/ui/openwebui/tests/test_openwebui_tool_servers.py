from unittest.mock import Mock

import httpx2
import pytest

from lgos_openwebui.functions.generic.gateway import gateway_config
from lgos_openwebui.tool_servers import (
    MCP_GATEWAY_ID,
    PUBLIC_READ_GRANT,
    mcp_gateway_connection,
    sync_mcp_gateway,
)


def _response(data: object) -> httpx2.Response:
    return httpx2.Response(
        200,
        json=data,
        request=httpx2.Request("GET", "http://openwebui.test/api"),
    )


@pytest.mark.parametrize(
    ("gateway_type", "url"),
    [
        ("litellm", "https://gateway.example/mcp/"),
        ("bifrost", "https://gateway.example/mcp"),
    ],
)
def test_connection_uses_each_gateways_native_mcp_contract(
    gateway_type: str,
    url: str,
) -> None:
    connection = mcp_gateway_connection(
        gateway_config(gateway_type, "https://gateway.example/"),
        "secret",
    )

    assert connection["url"] == url
    assert connection["auth_type"] == "bearer"
    assert connection["key"] == "secret"
    assert connection["config"]["function_name_filter_list"] == []
    assert connection["config"]["access_grants"] == [PUBLIC_READ_GRANT]


def test_sync_preserves_unrelated_native_tool_servers() -> None:
    unrelated = {"info": {"id": "other"}, "url": "https://other.example/mcp"}
    client = Mock()
    client.get.return_value = _response({"TOOL_SERVER_CONNECTIONS": [unrelated]})
    client.post.return_value = _response({})

    action = sync_mcp_gateway(
        client,
        gateway=gateway_config("litellm", "http://lgos-litellm:4000"),
        api_key="secret",
    )

    assert action == "created"
    payload = client.post.call_args.kwargs["json"]["TOOL_SERVER_CONNECTIONS"]
    assert payload[0] == unrelated
    assert payload[1]["info"]["id"] == MCP_GATEWAY_ID


def test_sync_skips_an_unchanged_connection() -> None:
    connection = mcp_gateway_connection(
        gateway_config("litellm", "http://lgos-litellm:4000"),
        "secret",
    )
    client = Mock()
    client.get.return_value = _response({"TOOL_SERVER_CONNECTIONS": [connection]})

    action = sync_mcp_gateway(
        client,
        gateway=gateway_config("litellm", "http://lgos-litellm:4000"),
        api_key="secret",
    )

    assert action == "unchanged"
    client.post.assert_not_called()
