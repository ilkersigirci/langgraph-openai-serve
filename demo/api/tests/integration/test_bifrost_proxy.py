import os

import pytest
from openai import AsyncOpenAI

BIFROST_BASE_URL = os.getenv("DEMO_TEST_BIFROST_BASE_URL")
BIFROST_CATALOG_BASE_URL = os.getenv("DEMO_TEST_BIFROST_CATALOG_BASE_URL")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.skipif(
        BIFROST_BASE_URL is None or BIFROST_CATALOG_BASE_URL is None,
        reason="set the Bifrost pass-through and catalog test URLs",
    ),
]


async def test_bifrost_catalog_and_passthrough_preserve_lgos() -> None:
    assert BIFROST_BASE_URL is not None
    assert BIFROST_CATALOG_BASE_URL is not None

    async with (
        AsyncOpenAI(
            base_url=BIFROST_CATALOG_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as catalog,
        AsyncOpenAI(
            base_url=BIFROST_BASE_URL,
            api_key="DUMMY",
            max_retries=0,
            timeout=10.0,
        ) as passthrough,
    ):
        catalog_models = await catalog.models.list()
        model_ids = {
            model.id
            for model in catalog_models.data
            if model.owned_by == "langgraph-openai-serve"
        }
        assert {"lgos-a/simple-graph", "lgos-b/simple-graph"} <= model_ids

        providers = sorted({model_id.split("/", 1)[0] for model_id in model_ids})
        for provider in providers:
            extra_headers = {"x-model-provider": provider}
            models = {
                model.id
                for model in (
                    await passthrough.models.list(extra_headers=extra_headers)
                ).data
            }
            assert "simple-graph" in models

            model = await passthrough.models.retrieve(
                "simple-graph",
                extra_headers=extra_headers,
            )
            extension = (model.model_extra or {})["langgraph_openai_serve"]
            assert extension["client_settings"]["schema_version"] == 1

            event_model = await passthrough.models.retrieve(
                "custom-event-showcase",
                extra_headers=extra_headers,
            )
            event_extension = (event_model.model_extra or {})["langgraph_openai_serve"]
            assert "client_events" in event_extension["features"]

            response = await passthrough.chat.completions.create(
                model="custom-input-output-context",
                messages=[{"role": "user", "content": "Show me custom schemas."}],
                user="demo-user",
                extra_headers=extra_headers,
            )
            assert response.choices[0].message.content == (
                "demo-user asked: Show me custom schemas."
            )

            stream = await passthrough.chat.completions.create(
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
