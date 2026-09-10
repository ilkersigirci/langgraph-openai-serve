import pytest


@pytest.fixture
def gateway_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply valid gateway settings without using the developer's environment."""
    monkeypatch.setenv("OPENAI_GATEWAY_TYPE", "litellm")
    monkeypatch.setenv("OPENAI_GATEWAY_BASE_URL", "http://lgos-litellm:4000")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "test-api-key")


@pytest.fixture
def anyio_backend() -> str:
    """Run the Open WebUI test suite on its supported async backend."""
    return "asyncio"
