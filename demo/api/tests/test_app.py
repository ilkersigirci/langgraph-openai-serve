from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
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
    "file-input",
    "interruptible-approval",
    "lgos-rag",
    "persistent-plot-agent",
    "multi-node-streaming",
    "simple-graph",
    "simple-graph-external-tools",
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
    features = {
        model.id: (model.model_extra or {})["langgraph_openai_serve"]["features"]
        for model in response.data
    }
    assert features["file-input"] == ["file_inputs"]

    interrupt_model = await openai_client.models.retrieve("interruptible-approval")
    extension = (interrupt_model.model_extra or {})["langgraph_openai_serve"]
    assert extension == {
        "schema_version": 1,
        "description": descriptions["interruptible-approval"],
        "features": ["interrupts"],
    }

    for model_id in ("complex-subgraphs", "custom-event-showcase", "status-events"):
        model = await openai_client.models.retrieve(model_id)
        extension = (model.model_extra or {})["langgraph_openai_serve"]
        assert extension == {
            "schema_version": 1,
            "description": descriptions[model_id],
            "features": ["client_events"],
        }

    plot_model = await openai_client.models.retrieve("persistent-plot-agent")
    plot_extension = (plot_model.model_extra or {})["langgraph_openai_serve"]
    assert plot_extension["features"] == []
    assert plot_extension["client_settings"]["defaults"] == {
        "chart_type": "bar",
        "currency": "USD",
        "show_legend": True,
    }
    assert plot_extension["client_settings"]["json_schema"]["properties"]["chart_type"][
        "enum"
    ] == ["bar", "line"]


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
    request = ResponseCreateRequest(
        model="simple-graph",
        input="Question",
        metadata=metadata,
    )

    graph_config = demo_app.state.graph_registry.get_graph("simple-graph")
    graph = await graph_config.resolve_graph()

    graph_request, _, _ = decode_responses_request(request)

    assert await graph_config.build_context(graph_request, graph) == expected_context


async def test_custom_io_demo_works_through_openai_client(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.responses.create(
        store=False,
        model="custom-input-output-context",
        input=[{"role": "user", "content": "Show me custom schemas."}],
        user="demo-user",
    )

    assert response.output_text == ("demo-user asked: Show me custom schemas.")


async def test_file_input_demo_prompts_for_an_attachment(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.responses.create(
        store=False,
        model="file-input",
        input=[{"role": "user", "content": "Summarize my file."}],
    )

    assert response.output_text == "Attach a file and try again."


async def test_complex_subgraphs_preserve_streaming_parity(
    openai_client: AsyncOpenAI,
) -> None:
    complete = await openai_client.responses.create(
        store=False,
        model="complex-subgraphs",
        input=[{"role": "user", "content": "Show nested subgraph routing docs."}],
    )
    stream = await openai_client.responses.create(
        store=False,
        model="complex-subgraphs",
        input=[{"role": "user", "content": "Show nested subgraph routing docs."}],
        stream=True,
    )

    phases = {}
    final_deltas = []
    async for event in stream:
        if event.type == "response.output_item.added" and event.item.type == "message":
            phases[event.output_index] = event.item.phase
        elif (
            event.type == "response.output_text.delta"
            and phases.get(event.output_index) == "final_answer"
        ):
            final_deltas.append(event.delta)
    streamed = "".join(final_deltas)

    assert streamed == complete.output_text


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
        assert demo_app.state.run_coordinator is coordinator
        assert demo_app.state.persistent_plot_agent.store is runtime.store

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
