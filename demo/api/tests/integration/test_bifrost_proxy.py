import os

import pytest
from openai import AsyncOpenAI

BIFROST_BASE_URL = os.getenv("DEMO_TEST_BIFROST_BASE_URL")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.skipif(
        BIFROST_BASE_URL is None,
        reason="set DEMO_TEST_BIFROST_BASE_URL to test through Bifrost",
    ),
]


async def test_bifrost_routes_two_lgos_apis_through_one_openai_client() -> None:
    assert BIFROST_BASE_URL is not None

    async with AsyncOpenAI(
        base_url=BIFROST_BASE_URL,
        api_key="DUMMY",
        max_retries=0,
        timeout=10.0,
    ) as client:
        for provider in ("lgos-a", "lgos-b"):
            extra_headers = {"x-model-provider": provider}
            models = {
                model.id
                for model in (
                    await client.models.list(extra_headers=extra_headers)
                ).data
            }
            assert "simple-graph" in models

            model = await client.models.retrieve(
                "simple-graph",
                extra_headers=extra_headers,
            )
            extension = (model.model_extra or {})["langgraph_openai_serve"]
            assert extension["client_settings"]["schema_version"] == 1

            event_model = await client.models.retrieve(
                "custom-event-showcase",
                extra_headers=extra_headers,
            )
            event_extension = (event_model.model_extra or {})["langgraph_openai_serve"]
            assert "client_events" in event_extension["features"]

            response = await client.chat.completions.create(
                model="custom-input-output-context",
                messages=[{"role": "user", "content": "Show me custom schemas."}],
                user="demo-user",
                extra_headers=extra_headers,
            )
            assert response.choices[0].message.content == (
                "demo-user asked: Show me custom schemas."
            )

            stream = await client.chat.completions.create(
                model="custom-event-showcase",
                messages=[{"role": "user", "content": "Build the report."}],
                metadata={"langgraph_stream_events": "v1"},
                stream=True,
                extra_headers=extra_headers,
            )
            event_types = []
            async for chunk in stream:
                extension = (chunk.model_extra or {}).get("langgraph_openai_serve", {})
                event = extension.get("event", {})
                if event_type := event.get("type"):
                    event_types.append(event_type)

            assert event_types == ["progress", "progress", "progress", "artifact"]
