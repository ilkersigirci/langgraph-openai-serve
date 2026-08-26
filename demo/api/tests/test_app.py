from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    Role,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from openai import AsyncOpenAI

from lgos_demo_api import app as app_module
from lgos_demo_api.checkpointer import PostgresRuntime
from lgos_demo_api.graphs.simple import SimpleContext

DOCUMENTED_MODEL_IDS = {
    "advanced-mcp-tools",
    "citation-events",
    "complex-subgraphs",
    "custom-event-showcase",
    "custom-input-output-context",
    "interruptible-approval",
    "lgos-rag",
    "persistent-plot",
    "simple-graph",
    "status-events",
}
CLIENT_SETTINGS_SCHEMA_VERSION = 1


@pytest.fixture
def demo_app() -> FastAPI:
    return app_module.create_custom_app()


@pytest.fixture
async def openai_client(demo_app: FastAPI) -> AsyncIterator[AsyncOpenAI]:
    async with (
        AsyncClient(
            transport=ASGITransport(app=demo_app),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as openai_client,
    ):
        yield openai_client


async def test_app_lists_exactly_the_documented_models(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert {model.id for model in response.data} == DOCUMENTED_MODEL_IDS
    descriptions = {
        model.id: (model.model_extra or {})["langgraph_openai_serve"]["description"]
        for model in response.data
    }
    assert all(description.strip() for description in descriptions.values())

    interrupt_model = await openai_client.models.retrieve("interruptible-approval")
    extension = (interrupt_model.model_extra or {})["langgraph_openai_serve"]
    assert extension == {
        "schema_version": 1,
        "description": descriptions["interruptible-approval"],
        "features": ["interrupts"],
    }

    for model_id in ("custom-event-showcase", "persistent-plot", "status-events"):
        model = await openai_client.models.retrieve(model_id)
        extension = (model.model_extra or {})["langgraph_openai_serve"]
        assert extension == {
            "schema_version": 1,
            "description": descriptions[model_id],
            "features": ["client_events"],
        }


async def test_cors_exposes_request_id(demo_app: FastAPI) -> None:
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/v1/health",
            headers={"Origin": "https://client.example"},
        )

    assert response.headers["access-control-expose-headers"] == "X-Request-ID"
    assert response.headers["x-request-id"]


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


async def test_lifespan_installs_shared_postgres_runtime(
    demo_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    coordinator = InMemoryRunCoordinator()

    runtime = PostgresRuntime(
        checkpointer=sqlite_checkpointer,  # type: ignore[arg-type]
        store=InMemoryStore(),  # type: ignore[arg-type]
        run_coordinator=coordinator,  # type: ignore[arg-type]
    )

    @asynccontextmanager
    async def postgres_runtime(postgres_uri: str):
        assert postgres_uri == app_module.settings.POSTGRES_URI
        yield runtime

    runtime_factory = Mock(wraps=postgres_runtime)
    monkeypatch.setattr(app_module, "postgres_runtime", runtime_factory)

    async with app_module.lifespan(demo_app):
        assert demo_app.state.interruptible_graph.checkpointer is sqlite_checkpointer
        assert demo_app.state.interruptible_run_coordinator is coordinator
        assert demo_app.state.persistent_plot_graph.store is runtime.store

        config = demo_app.state.graph_registry.get_graph("interruptible-approval")
        assert config.run_coordinator is not None
        async with config.run_coordinator("thread-1"):
            pass

    runtime_factory.assert_called_once_with(app_module.settings.POSTGRES_URI)


def test_main_leaves_access_logging_to_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)

    app_module.main()

    run.assert_called_once()
    assert run.call_args.kwargs["access_log"] is False
