import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the supported backend."""
    return "asyncio"
