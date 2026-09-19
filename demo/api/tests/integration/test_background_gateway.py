import asyncio
import os
import time
import uuid

import pytest
from openai import AsyncOpenAI, BadRequestError

GATEWAY_BASE_URL = os.getenv("DEMO_TEST_BACKGROUND_GATEWAY_BASE_URL")
GATEWAY_API_KEY = os.getenv("OPENAI_GATEWAY_API_KEY", "DUMMY")
MODEL = os.getenv(
    "DEMO_TEST_BACKGROUND_GATEWAY_MODEL",
    "background-report-agent",
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        GATEWAY_BASE_URL is None,
        reason="set the dedicated background gateway test URL",
    ),
]


def _client() -> AsyncOpenAI:
    assert GATEWAY_BASE_URL is not None
    return AsyncOpenAI(
        base_url=GATEWAY_BASE_URL,
        api_key=GATEWAY_API_KEY,
        max_retries=0,
        timeout=30.0,
    )


async def test_gateway_polls_saved_response_id_with_a_new_client() -> None:
    async with _client() as client:
        created = await client.responses.create(
            model=MODEL,
            input="Write a two-sentence report about durable background execution.",
            background=True,
            stream=False,
            store=True,
            metadata={"lgos_run_id": str(uuid.uuid4())},
        )

    assert created.status in {"queued", "in_progress"}
    deadline = time.monotonic() + 60
    response = created
    async with _client() as restarted_client:
        while response.status in {"queued", "in_progress"}:
            assert time.monotonic() < deadline
            await asyncio.sleep(1)
            response = await restarted_client.responses.retrieve(created.id)

    assert response.id == created.id
    assert response.status == "completed"
    assert response.output_text


async def test_gateway_cancels_with_only_the_saved_response_id() -> None:
    async with _client() as client:
        created = await client.responses.create(
            model=MODEL,
            input="Cancel this background report.",
            background=True,
            stream=False,
            store=True,
            metadata={"lgos_run_id": str(uuid.uuid4())},
        )

    async with _client() as restarted_client:
        cancelled = await restarted_client.responses.cancel(created.id)
        retrieved = await restarted_client.responses.retrieve(created.id)

    assert cancelled.id == retrieved.id == created.id
    assert cancelled.status == retrieved.status == "cancelled"


async def test_gateway_preserves_polling_only_validation() -> None:
    async with _client() as client:
        with pytest.raises(BadRequestError) as create_error:
            await client.responses.create(
                model=MODEL,
                input="Streaming is unsupported.",
                background=True,
                stream=True,
                metadata={"lgos_run_id": str(uuid.uuid4())},
            )
        assert create_error.value.status_code == 400

        created = await client.responses.create(
            model=MODEL,
            input="Validate retrieval options.",
            background=True,
            stream=False,
            metadata={"lgos_run_id": str(uuid.uuid4())},
        )
        with pytest.raises(BadRequestError) as stream_error:
            await client.responses.retrieve(
                created.id,
                extra_query={"stream": "true"},
            )
        with pytest.raises(BadRequestError) as cursor_error:
            await client.responses.retrieve(
                created.id,
                extra_query={"starting_after": "0"},
            )

    assert stream_error.value.status_code == 400
    assert cursor_error.value.status_code == 400
