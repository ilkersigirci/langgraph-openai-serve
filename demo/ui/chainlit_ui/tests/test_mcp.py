import pytest

from lgos_chainlit import mcp as mcp_module
from lgos_chainlit.gateway import gateway_config


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
