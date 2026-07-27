import os

import pytest
from openai import AsyncOpenAI

BIFROST_CATALOG_BASE_URL = os.getenv("DEMO_TEST_BIFROST_CATALOG_BASE_URL")
BIFROST_PASSTHROUGH_BASE_URL = os.getenv(
    "DEMO_TEST_BIFROST_PASSTHROUGH_BASE_URL",
    "http://localhost:3000/openai_passthrough/v1",
)
LGOS_PROVIDERS = ("lgos-a", "lgos-b")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.skipif(
        BIFROST_CATALOG_BASE_URL is None,
        reason="set DEMO_TEST_BIFROST_CATALOG_BASE_URL to test through Bifrost",
    ),
]


async def test_bifrost_combines_both_lgos_servers() -> None:
    assert BIFROST_CATALOG_BASE_URL is not None

    async with (
        AsyncOpenAI(
            base_url=BIFROST_CATALOG_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as catalog_client,
        AsyncOpenAI(
            base_url=BIFROST_PASSTHROUGH_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as passthrough_client,
    ):
        models = {model.id for model in (await catalog_client.models.list()).data}
        for provider in LGOS_PROVIDERS:
            assert f"{provider}/simple-graph" in models

            headers = {"x-model-provider": provider}
            model = await passthrough_client.models.retrieve(
                "simple-graph",
                extra_headers=headers,
            )
            extension = (model.model_extra or {})["langgraph_openai_serve"]
            assert extension["client_settings"]["schema_version"] == 1

            response = await passthrough_client.chat.completions.create(
                model="custom-input-output-context",
                messages=[{"role": "user", "content": "Show me custom schemas."}],
                user="demo-user",
                extra_headers=headers,
            )
            assert response.choices[0].message.content == (
                "demo-user asked: Show me custom schemas."
            )

            stream = await passthrough_client.chat.completions.create(
                model="custom-event-showcase",
                messages=[{"role": "user", "content": "Build the report."}],
                metadata={"langgraph_stream_events": "v1"},
                stream=True,
                extra_headers=headers,
            )
            event_types = []
            async for chunk in stream:
                extension = (chunk.model_extra or {}).get("langgraph_openai_serve", {})
                event = extension.get("event", {})
                if event_type := event.get("type"):
                    event_types.append(event_type)

            assert event_types == ["progress", "progress", "progress", "artifact"]
