"""Request correlation and application error logging behavior."""

import logging
import uuid

import pytest
from anyio import Event, create_task_group
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette import status

from langgraph_openai_serve import GraphConfig, GraphFeature, GraphRegistry
from langgraph_openai_serve.api.middleware import RequestContextMiddleware
from langgraph_openai_serve.core.logging import bind_log_context, get_logger
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.openai_server import LanggraphOpenaiServe

_TEST_LOGGER = get_logger("langgraph_openai_serve.tests.request_context")
_UUID4_VERSION = 4


def _records(caplog, event: str):
    return [record for record in caplog.records if record.getMessage() == event]


async def test_missing_request_id_is_generated(client: AsyncClient) -> None:
    response = await client.get("/v1/models")

    request_id = response.headers["x-request-id"]
    assert uuid.UUID(request_id).version == _UUID4_VERSION


async def test_incoming_request_id_is_preserved(client: AsyncClient) -> None:
    response = await client.get(
        "/v1/models",
        headers={"X-Request-ID": " upstream-request-123 "},
    )

    assert response.headers["x-request-id"] == "upstream-request-123"


async def test_unusable_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get(
        "/v1/models",
        headers={"X-Request-ID": "x" * 129},
    )

    request_id = response.headers["x-request-id"]
    assert request_id != "x" * 129
    assert uuid.UUID(request_id).version == _UUID4_VERSION


async def test_request_context_is_added_to_lgos_logs(caplog) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")

    async def app(scope, _receive, send) -> None:
        bind_log_context(model="test", stream=False)
        _TEST_LOGGER.info("test.request")
        await send(
            {
                "type": "http.response.start",
                "status": status.HTTP_200_OK,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/models",
        "headers": [(b"x-request-id", b"request-123")],
    }

    async def send(_message) -> None:
        return None

    await RequestContextMiddleware(app)(scope, dict, send)

    record = next(
        record for record in caplog.records if record.getMessage() == "test.request"
    )
    assert record.request_id == "request-123"
    assert record.model == "test"
    assert record.stream is False


async def test_escaping_exception_is_logged_and_reraised(caplog) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")
    error_message = "boom"

    async def app(_scope, _receive, _send) -> None:
        raise RuntimeError(error_message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/failure",
        "headers": [(b"x-request-id", b"failure-request")],
    }

    with pytest.raises(RuntimeError, match="boom"):
        await RequestContextMiddleware(app)(scope, dict, dict)

    records = _records(caplog, "http.request.failed")
    assert len(records) == 1
    assert records[0].request_id == "failure-request"
    assert records[0].http_method == "GET"
    assert records[0].http_path == "/failure"
    assert records[0].error_type == "RuntimeError"
    assert records[0].exc_info is not None

    _TEST_LOGGER.info("test.after_failure")
    after_failure = next(
        record
        for record in caplog.records
        if record.getMessage() == "test.after_failure"
    )
    assert not hasattr(after_failure, "request_id")


async def test_handled_server_error_is_logged(
    message_graph,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")
    registry = GraphRegistry(
        registry={
            "broken": GraphConfig(
                graph=message_graph,
                description="Broken graph",
                features={GraphFeature.INTERRUPTS},
                run_coordinator=InMemoryRunCoordinator(),
            )
        }
    )
    app = LanggraphOpenaiServe(graphs=registry).bind_openai_api(prefix="/v1").app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "server-error"},
            json={
                "model": "broken",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.headers["x-request-id"] == "server-error"
    records = _records(caplog, "http.request.failed")
    assert len(records) == 1
    assert records[0].request_id == "server-error"
    assert records[0].status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert records[0].exc_info is not None


async def test_unhandled_error_response_has_request_id(
    graph_registry: GraphRegistry,
    caplog,
) -> None:
    caplog.set_level(logging.ERROR, logger="langgraph_openai_serve")

    def failing_checkpoint_scope(_request):
        msg = "Checkpoint scope failed"
        raise RuntimeError(msg)

    app = (
        LanggraphOpenaiServe(
            graphs=graph_registry,
            checkpoint_scope=failing_checkpoint_scope,
        )
        .bind_openai_api(prefix="/v1")
        .app
    )

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "unhandled-error"},
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.headers["x-request-id"] == "unhandled-error"
    records = _records(caplog, "http.request.failed")
    assert len(records) == 1
    assert records[0].request_id == "unhandled-error"
    assert records[0].error_type == "RuntimeError"


async def test_host_routes_are_not_wrapped_by_lgos_middleware(
    graph_registry: GraphRegistry,
) -> None:
    app = FastAPI()

    @app.get("/host-route")
    async def host_route() -> dict[str, bool]:
        return {"ok": True}

    LanggraphOpenaiServe(app=app, graphs=graph_registry).bind_openai_api(prefix="/v1")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        host_response = await client.get("/host-route")
        lgos_response = await client.get("/v1/models")

    assert "x-request-id" not in host_response.headers
    assert lgos_response.headers.get("x-request-id")


async def test_concurrent_request_contexts_do_not_cross_contaminate(caplog) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")
    entered = {"model-a": Event(), "model-b": Event()}
    release = Event()

    async def app(scope, _receive, send) -> None:
        model = scope["path"].removeprefix("/")
        bind_log_context(model=model, stream=False)
        _TEST_LOGGER.info("test.request.started")
        entered[model].set()
        await release.wait()
        _TEST_LOGGER.info("test.request.finished")
        await send(
            {
                "type": "http.response.start",
                "status": status.HTTP_200_OK,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    transport = ASGITransport(app=RequestContextMiddleware(app))
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        create_task_group() as task_group,
    ):

        async def get_request(path: str, request_id: str) -> None:
            await client.get(path, headers={"X-Request-ID": request_id})

        task_group.start_soon(get_request, "/model-a", "request-a")
        task_group.start_soon(get_request, "/model-b", "request-b")
        await entered["model-a"].wait()
        await entered["model-b"].wait()
        release.set()

    request_records = [
        record for record in caplog.records if hasattr(record, "request_id")
    ]
    assert request_records
    expected_models = {"request-a": "model-a", "request-b": "model-b"}
    assert all(
        record.model == expected_models[record.request_id] for record in request_records
    )


def test_non_request_log_has_no_request_fields(caplog) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")

    _TEST_LOGGER.info("test.event")

    record = next(
        record for record in caplog.records if record.getMessage() == "test.event"
    )
    assert not hasattr(record, "request_id")
