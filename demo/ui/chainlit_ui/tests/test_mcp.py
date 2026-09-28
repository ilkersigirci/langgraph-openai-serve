import pytest

from lgos_chainlit import mcp as mcp_module
from lgos_chainlit.gateway import gateway_config


@pytest.mark.parametrize("gateway_type", ["litellm", "bifrost"])
def test_gateway_config_builds_the_aggregate_mcp_endpoint(gateway_type: str) -> None:
    server = mcp_module.mcp_gateway_config(
        gateway_config(gateway_type, "https://gateway.example/"),
        "secret",
    )

    assert (server.name, server.url) == (
        mcp_module.MCP_GATEWAY_NAME,
        "https://gateway.example/mcp",
    )
    assert server.headers == {"Authorization": "Bearer secret"}
