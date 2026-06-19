from collections.abc import AsyncGenerator, Generator
from typing import Annotated, TypedDict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LangchainOpenaiApiServe,
)

_BASE_URL = "http://test"
_TIMEOUT = 2.0


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def message_graph():
    model = FakeListChatModel(responses=["hello"])

    async def generate(state: MessageState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    return (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )


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
