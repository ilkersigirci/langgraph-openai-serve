from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from tests.graph.support.message import make_message_graph as build_message_graph

_BASE_URL = "http://test"
_TIMEOUT = 2.0


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run the package test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
def message_graph():
    return build_message_graph()


@pytest.fixture
def graph_registry(message_graph) -> GraphRegistry:
    return GraphRegistry(
        registry={
            "test": GraphConfig(
                graph=message_graph,
                streamable_node_names=["generate"],
            )
        }
    )


@pytest.fixture
def fastapi_app(graph_registry: GraphRegistry) -> FastAPI:
    return (
        LanggraphOpenaiServe(
            graphs=graph_registry,
        )
        .bind_openai_api()
        .app
    )


@pytest.fixture
async def client(fastapi_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(
        transport=transport,
        base_url=_BASE_URL,
        timeout=_TIMEOUT,
    ) as async_client:
        yield async_client


@pytest.fixture
async def openai_client(
    client: AsyncClient,
) -> AsyncGenerator[AsyncOpenAI, None]:
    openai_client = AsyncOpenAI(
        api_key="test",
        base_url=f"{_BASE_URL}/v1",
        http_client=client,
        max_retries=0,
    )
    yield openai_client
    await openai_client.close()
