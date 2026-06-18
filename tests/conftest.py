from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Annotated, TypedDict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
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

_BASE_URL = "http://testserver/v1"
_API_KEY = "test"


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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
    """Return the default graph registry used by API tests."""
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
    """Create the FastAPI application with OpenAI chat routes bound."""
    return (
        LangchainOpenaiApiServe(graphs=graph_registry).bind_openai_chat_completion().app
    )


@pytest.fixture
def openai_client(fastapi_app: FastAPI) -> Iterator[OpenAI]:
    """Create an official OpenAI client backed by FastAPI's test client."""
    with TestClient(fastapi_app) as http_client:
        yield OpenAI(
            api_key=_API_KEY,
            base_url=_BASE_URL,
            http_client=http_client,
        )


@pytest.fixture
def openai_client_factory() -> Callable[
    [GraphRegistry], AbstractContextManager[OpenAI]
]:
    """Create OpenAI test clients for custom graph registries."""

    @contextmanager
    def create(graph_registry: GraphRegistry) -> Iterator[OpenAI]:
        app = (
            LangchainOpenaiApiServe(graphs=graph_registry)
            .bind_openai_chat_completion()
            .app
        )
        with TestClient(app) as http_client:
            yield OpenAI(
                api_key=_API_KEY,
                base_url=_BASE_URL,
                http_client=http_client,
            )

    return create
