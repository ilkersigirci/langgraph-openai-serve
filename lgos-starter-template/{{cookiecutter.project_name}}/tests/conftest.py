from collections.abc import AsyncIterator

import pytest
from openai import AsyncOpenAI

from tests.support import create_test_app, started


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def openai_client() -> AsyncIterator[AsyncOpenAI]:
    async with started(create_test_app()) as client:
        yield client
