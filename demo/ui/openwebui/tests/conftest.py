import sys
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

GATEWAY_ROOT = "http://lgos-litellm:4000"
OpenWebUIConnection = Callable[..., None]


@pytest.fixture
def openwebui_connection(monkeypatch: pytest.MonkeyPatch) -> OpenWebUIConnection:
    """Set Open WebUI's own OpenAI connection, whose auth type the Pipe follows."""

    def connect(auth_type: str, url: str = f"{GATEWAY_ROOT}/v1") -> None:
        runtime_config = (True, [url], [""], {"0": {"auth_type": auth_type}})
        openai_router = SimpleNamespace(
            get_openai_runtime_config=AsyncMock(return_value=runtime_config)
        )
        monkeypatch.setitem(sys.modules, "open_webui.routers.openai", openai_router)

    connect("bearer")
    return connect


@pytest.fixture
def gateway_environment(
    monkeypatch: pytest.MonkeyPatch, openwebui_connection: OpenWebUIConnection
) -> None:
    """Supply valid gateway settings without using the developer's environment."""
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "litellm")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", GATEWAY_ROOT)
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "test-api-key")


@pytest.fixture
def anyio_backend() -> str:
    """Run the Open WebUI test suite on its supported async backend."""
    return "asyncio"
