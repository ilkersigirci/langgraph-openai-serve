import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run the Chainlit test suite on its supported async backend."""
    return "asyncio"
