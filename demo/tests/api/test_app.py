import importlib
import logging.config
from collections.abc import AsyncIterator

import pytest
from demo.api.graphs.simple import SimpleContext
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    Role,
)

DOCUMENTED_MODEL_IDS = {
    "advanced-mcp-tools",
    "citation-events",
    "complex-subgraphs",
    "custom-input-output-context",
    "interruptible-approval",
    "lgos-rag",
    "simple-graph",
}
CLIENT_SETTINGS_SCHEMA_VERSION = 1


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
    assert all(not model.model_extra for model in response.data)

    interrupt_model = await openai_client.models.retrieve("interruptible-approval")
    extension = (interrupt_model.model_extra or {})["langgraph_openai_serve"]
    assert extension == {"schema_version": 1, "features": ["interrupts"]}


@pytest.mark.anyio
async def test_simple_model_retrieval_exposes_runtime_settings(
    openai_client: AsyncOpenAI,
) -> None:
    model = await openai_client.models.retrieve("simple-graph")

    extension = (model.model_extra or {})["langgraph_openai_serve"]
    client_settings = extension["client_settings"]
    assert client_settings["schema_version"] == CLIENT_SETTINGS_SCHEMA_VERSION
    assert client_settings["defaults"] == {
        "use_history": False,
        "audience": "general",
    }
    assert client_settings["json_schema"]["properties"]["audience"]["enum"] == [
        "general",
        "beginner",
        "expert",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("metadata", "expected_context"),
    [
        (None, SimpleContext()),
        (
            {"langgraph_runtime_settings": '{"use_history":true}'},
            SimpleContext(use_history=True),
        ),
        (
            {"langgraph_runtime_settings": '{"audience":"expert"}'},
            SimpleContext(audience="expert"),
        ),
    ],
)
async def test_simple_model_builds_its_runtime_context(
    demo_app: FastAPI,
    metadata: dict[str, str] | None,
    expected_context: SimpleContext,
) -> None:
    request = ChatCompletionRequest(
        model="simple-graph",
        messages=[ChatCompletionRequestMessage(role=Role.USER, content="Question")],
        metadata=metadata,
    )

    graph_config = demo_app.state.graph_registry.get_graph("simple-graph")
    graph = await graph_config.resolve_graph()

    assert await graph_config.build_context(request, graph) == expected_context


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
