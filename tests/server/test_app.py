from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import anyio
import pytest
from httpx2 import ASGITransport, AsyncClient

from langgraph_openai_serve import GraphConfig, GraphRegistry
from langgraph_openai_serve.server import ServerResources, create_app, runtime
from tests.graph.support.message import make_message_graph
from tests.server.support import create_registry, server_settings, started


async def test_interrupt_graph_resumes_with_server_resources() -> None:
    app = create_app(create_registry, settings=server_settings())
    async with started(app) as client:
        paused = await client.responses.create(
            model="approval", input="Publish the report", store=False
        )
        (call,) = [item for item in paused.output if item.type == "function_call"]
        resumed = await client.responses.create(
            model="approval",
            previous_response_id=paused.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "approve",
                }
            ],
            store=False,
        )
    assert resumed.output_text == "resumed:approve"


async def test_factory_context_stays_open_for_the_app_lifetime() -> None:
    states: list[str] = []

    @asynccontextmanager
    async def open_graphs(
        resources: ServerResources,
    ) -> AsyncGenerator[GraphRegistry, None]:
        states.append("opened")
        try:
            yield create_registry(resources)
        finally:
            states.append("closed")

    app = create_app(open_graphs, settings=server_settings())
    async with started(app) as client:
        response = await client.responses.create(
            model="chat", input="Hello", store=False
        )
        assert states == ["opened"]
    assert response.output_text == "hello"
    assert states == ["opened", "closed"]


async def test_memory_background_completes_a_polled_response() -> None:
    app = create_app(create_registry, settings=server_settings(BACKGROUND="memory"))
    async with started(app) as client:
        response = await client.responses.create(
            model="chat",
            input="Hello",
            background=True,
            store=True,
            extra_headers={"Idempotency-Key": "server-memory-background"},
        )
        with anyio.fail_after(5):
            while response.status in {"queued", "in_progress"}:
                await anyio.lowlevel.checkpoint()
                response = await client.responses.retrieve(response.id)
    assert response.status == "completed"
    assert response.output_text == "hello"


@pytest.mark.parametrize(
    ("origin", "status_code", "allowed_origin"),
    [
        ("https://app.example.com", 200, "https://app.example.com"),
        ("https://untrusted.example.com", 400, None),
    ],
)
async def test_cors_allows_only_configured_origins(
    origin: str, status_code: int, allowed_origin: str | None
) -> None:
    app = create_app(
        create_registry,
        settings=server_settings(CORS_ORIGINS=["https://app.example.com"]),
    )
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        response = await http.options(
            "/v1/responses",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
    assert response.status_code == status_code
    assert response.headers.get("access-control-allow-origin") == allowed_origin


async def test_cors_exposes_the_request_id_to_browser_code() -> None:
    origin = "https://app.example.com"
    app = create_app(create_registry, settings=server_settings(CORS_ORIGINS=[origin]))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        response = await http.get("/v1/health", headers={"Origin": origin})

    # Browsers hide response headers from scripts unless CORS exposes them.
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-expose-headers"] == "X-Request-ID"
    assert response.headers["x-request-id"]


async def test_restarted_app_serves_the_newly_opened_registry() -> None:
    model_ids = iter(["first", "second"])

    def open_graphs(_resources: ServerResources) -> GraphRegistry:
        return GraphRegistry(
            graphs={
                next(model_ids): GraphConfig(
                    graph=make_message_graph(), description="Answer."
                )
            }
        )

    app = create_app(open_graphs, settings=server_settings())
    for expected in ("first", "second"):
        async with started(app) as client:
            models = await client.models.list()
        assert [model.id for model in models.data] == [expected]


@pytest.mark.parametrize(("interval", "sweeps"), [(5, 1), (0, 0)])
async def test_interrupt_expiry_sweeps_unless_disabled(
    monkeypatch: pytest.MonkeyPatch, interval: int, sweeps: int
) -> None:
    sweep = AsyncMock(return_value=0)
    monkeypatch.setattr(runtime, "delete_expired_interrupt_runs", sweep)
    app = create_app(
        create_registry,
        settings=server_settings(INTERRUPT_SWEEP_INTERVAL_MINUTES=interval),
    )

    async with app.router.lifespan_context(app):
        await anyio.wait_all_tasks_blocked()

    assert sweep.await_count == sweeps
