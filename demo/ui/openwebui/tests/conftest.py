import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run the Open WebUI test suite on its supported async backend."""
    return "asyncio"
