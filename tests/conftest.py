from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from tests.graph.support.message import make_message_graph as build_message_graph

_BASE_URL = "http://test"
_TIMEOUT = 2.0


@pytest.fixture
def anyio_backend() -> str:
    """Run the package test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
def message_graph():
    return build_message_graph()


@pytest.fixture
async def sqlite_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@pytest.fixture
def graph_registry(message_graph) -> GraphRegistry:
    return GraphRegistry(
        registry={
            "test": GraphConfig(
                graph=message_graph,
                description="DUMMY",
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
async def client(fastapi_app: FastAPI) -> AsyncIterator[AsyncClient]:
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
) -> AsyncIterator[AsyncOpenAI]:
    async with AsyncOpenAI(
        api_key="test",
        base_url=f"{_BASE_URL}/v1",
        http_client=client,
        max_retries=0,
    ) as openai_client:
        yield openai_client
