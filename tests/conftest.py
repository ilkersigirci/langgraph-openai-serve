from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import OpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LangchainOpenaiApiServe,
)
from tests.graph.support.message import make_message_graph as build_message_graph

_BASE_URL = "http://test"
_TIMEOUT = 2.0


@pytest.fixture(scope="session")
def anyio_backend() -> str:
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
def fastapi_app(graph_registry: GraphRegistry) -> Generator[FastAPI, None, None]:
    application = LangchainOpenaiApiServe(
        graphs=graph_registry,
    ).bind_openai_chat_completion().app
    yield application


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
def openai_client(fastapi_app: FastAPI) -> Generator[OpenAI, None, None]:
    with TestClient(fastapi_app) as http_client:
        yield OpenAI(
            api_key="test",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
