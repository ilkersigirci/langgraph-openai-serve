import os

import pytest
from openai import AsyncOpenAI

BIFROST_BASE_URL = os.getenv("DEMO_TEST_BIFROST_BASE_URL")
DIRECT_BASE_URL = os.getenv("DEMO_TEST_DIRECT_BASE_URL", "http://localhost:8000/v1")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.skipif(
        BIFROST_BASE_URL is None,
        reason="set DEMO_TEST_BIFROST_BASE_URL to test through Bifrost",
    ),
]


async def test_model_metadata_through_bifrost() -> None:
    assert BIFROST_BASE_URL is not None

    async with (
        AsyncOpenAI(
            base_url=DIRECT_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as direct_client,
        AsyncOpenAI(
            base_url=BIFROST_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as proxy_client,
    ):
        direct = await direct_client.models.retrieve("simple-graph")
        proxied = await proxy_client.models.retrieve("simple-graph")

    direct_extension = (direct.model_extra or {})["langgraph_openai_serve"]
    proxied_extension = (proxied.model_extra or {})["langgraph_openai_serve"]
    assert proxied_extension == direct_extension
