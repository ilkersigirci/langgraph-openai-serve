import importlib
import logging.config
from collections.abc import AsyncIterator

import pytest
from demo.api.graphs.simple import DEFAULT_SYSTEM_PROMPT, SimpleContext
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
    assert {
        model.id: (model.model_extra or {})["langgraph_openai_serve"]["features"]
        for model in response.data
        if (model.model_extra or {})["langgraph_openai_serve"]["features"]
    } == {"interruptible-approval": ["interrupts"]}
    assert all(
        (model.model_extra or {})["langgraph_openai_serve"]["schema_version"] == 1
        for model in response.data
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "use_history", "system_prompt"),
    [
        ("simple-graph-with-history", True, None),
        ("simple-graph-no-history", False, "Respond in Turkish."),
    ],
)
async def test_simple_models_build_their_runtime_context(
    demo_app: FastAPI,
    model: str,
    use_history: bool,
    system_prompt: str | None,
) -> None:
    request = ChatCompletionRequest(
        model=model,
        messages=[ChatCompletionRequestMessage(role=Role.USER, content="Question")],
        metadata={"system_prompt": system_prompt} if system_prompt else None,
    )

    graph_config = demo_app.state.graph_registry.get_graph(model)

    assert await graph_config.build_context(request) == SimpleContext(
        use_history=use_history,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )


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
