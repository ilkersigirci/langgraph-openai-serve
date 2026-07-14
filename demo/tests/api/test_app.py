import importlib
import logging.config
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI

DOCUMENTED_MODEL_IDS = {
    "advanced-mcp-tools",
    "citation-events",
    "complex-subgraphs",
    "custom-input-output-context",
    "interruptible-approval",
    "lgos-rag",
    "simple-graph-no-history",
    "simple-graph-with-history",
}


@pytest.fixture
def demo_app(
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    monkeypatch.setattr(logging.config, "dictConfig", lambda _: None)
    return importlib.import_module("demo.api.app").app


@pytest.fixture
async def openai_client(demo_app: FastAPI) -> AsyncIterator[AsyncOpenAI]:
    http_client = AsyncClient(
        transport=ASGITransport(app=demo_app),
        base_url="http://test",
    )
    openai_client = AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=http_client,
        max_retries=0,
    )
    yield openai_client
    await openai_client.close()


@pytest.mark.anyio
async def test_app_lists_exactly_the_documented_models(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert {model.id for model in response.data} == DOCUMENTED_MODEL_IDS


@pytest.mark.anyio
async def test_custom_io_demo_works_through_openai_client(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.chat.completions.create(
        model="custom-input-output-context",
        messages=[{"role": "user", "content": "Show me custom schemas."}],
        user="demo-user",
    )

    assert response.choices[0].message.content == (
        "demo-user asked: Show me custom schemas."
    )
