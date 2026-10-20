import sys
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

GATEWAY_ROOT = "http://lgos-litellm:4000"
OpenWebUIConnection = Callable[..., None]


@pytest.fixture
def openwebui_connection(monkeypatch: pytest.MonkeyPatch) -> OpenWebUIConnection:
    """Set Open WebUI's own OpenAI connection, which the Pipe authenticates like."""

    def connect(
        auth_type: str | None,
        url: str = f"{GATEWAY_ROOT}/v1",
        key: str = "test-api-key",
    ) -> None:
        runtime_config = (True, [url], [key], {"0": {"auth_type": auth_type}})
        openai_router = SimpleNamespace(
            get_openai_runtime_config=AsyncMock(return_value=runtime_config)
        )
        monkeypatch.setitem(sys.modules, "open_webui.routers.openai", openai_router)

    connect("bearer")
    return connect


@pytest.fixture
def anyio_backend() -> str:
    """Run the Open WebUI test suite on its supported async backend."""
    return "asyncio"
