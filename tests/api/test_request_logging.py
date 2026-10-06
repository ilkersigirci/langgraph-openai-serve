"""Request correlation and application error logging behavior."""

import logging
import uuid

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from opentelemetry.sdk.trace import TracerProvider
from starlette import status

from langgraph_openai_serve import (
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    RequestContextFilter,
)
from langgraph_openai_serve.api.middleware import RequestContextMiddleware
from langgraph_openai_serve.core.errors import configure_openai_error_handlers
from langgraph_openai_serve.core.logging import bind_log_context, get_logger
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from langgraph_openai_serve.openai_server import LanggraphOpenaiServe
from tests.graph.support.schemas import MessageState
from tests.graph.support.telemetry import TraceContextHandler

_TEST_LOGGER = get_logger("langgraph_openai_serve.tests.request_context")
_UUID4_VERSION = 4


class _TestRequestError(Exception):
    pass


def _records(caplog, event: str):
    return [record for record in caplog.records if record.getMessage() == event]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        pytest.param([], None, id="missing"),
        pytest.param(
            [(b"x-request-id", b" upstream-request-123 ")],
            "upstream-request-123",
            id="preserved",
        ),
        pytest.param([(b"x-request-id", b"x" * 129)], None, id="too-long"),
        pytest.param([(b"x-request-id", b"request-\x85forged")], None, id="unsafe"),
        pytest.param(
            [(b"x-request-id", b"request-one"), (b"x-request-id", b"request-two")],
            None,
            id="ambiguous",
        ),
    ],
)
async def test_request_id_is_preserved_or_generated(
    client: AsyncClient,
    headers,
    expected: str | None,
) -> None:
    response = await client.get("/v1/models", headers=headers)

    request_id = response.headers["x-request-id"]
    if expected is None:
        assert uuid.UUID(request_id).version == _UUID4_VERSION
    else:
        assert request_id == expected


async def test_request_context_is_added_to_lgos_logs(caplog) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")

    async def app(scope, _receive, send) -> None:
        bind_log_context(
            model="test",
            stream=False,
            operation_id="operation-123",
        )
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
    assert record.operation_id == "operation-123"


async def test_escaping_exception_is_reraised_and_context_is_reset(caplog) -> None:
    caplog.set_level(logging.INFO, logger="langgraph_openai_serve")
    error_message = "boom"

    async def app(_scope, _receive, _send) -> None:
        raise _TestRequestError(error_message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/failure",
        "headers": [(b"x-request-id", b"failure-request")],
    }

    with pytest.raises(_TestRequestError, match="boom"):
        await RequestContextMiddleware(app)(scope, dict, dict)

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
        graphs={
            # A factory is validated per request, so its error is a server error.
            "broken": GraphConfig(
                graph=lambda: message_graph,
                description="Broken graph",
                features={GraphFeature.INTERRUPTS},
            )
        },
        run_coordinator=InMemoryRunCoordinator(),
    )
    app = LanggraphOpenaiServe(registry=registry).bind_openai_api(prefix="/v1").app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/responses",
            headers={"X-Request-ID": "server-error"},
            json={
                "model": "broken",
                "input": "Hello",
            },
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.headers["x-request-id"] == "server-error"
    records = _records(caplog, "http.request.failed")
    assert len(records) == 1
    assert records[0].request_id == "server-error"
    assert (
        records[0].__dict__["http.response.status_code"]
        == status.HTTP_500_INTERNAL_SERVER_ERROR
    )
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
            registry=graph_registry,
            checkpoint_scope=failing_checkpoint_scope,
        )
        .bind_openai_api(prefix="/v1")
        .app
    )

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/responses",
            headers={"X-Request-ID": "unhandled-error"},
            json={
                "model": "test",
                "input": "Hello",
            },
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.headers["x-request-id"] == "unhandled-error"
    records = _records(caplog, "http.request.failed")
    assert len(records) == 1
    assert records[0].request_id == "unhandled-error"
    assert records[0].__dict__["error.type"] == "RuntimeError"


async def test_stream_failure_keeps_request_context_in_producer_task(
    caplog,
) -> None:
    caplog.set_level(logging.ERROR, logger="langgraph_openai_serve")

    async def fail(_state: MessageState) -> dict[str, object]:
        msg = "stream failed"
        raise RuntimeError(msg)

    graph = (
        StateGraph(MessageState)
        .add_node("fail", fail)
        .set_entry_point("fail")
        .set_finish_point("fail")
        .compile()
    )
    registry = GraphRegistry(
        graphs={
            "broken-stream": GraphConfig(
                graph=graph,
                description="Broken stream",
            )
        }
    )
    app = LanggraphOpenaiServe(registry=registry).bind_openai_api(prefix="/v1").app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "stream-error"},
            json={
                "model": "broken-stream",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == status.HTTP_200_OK
    records = _records(caplog, "chat_completion.stream_failed")
    assert len(records) == 1
    assert records[0].request_id == "stream-error"
    assert records[0].model == "broken-stream"
    assert records[0].stream is True


async def test_host_routes_are_not_wrapped_by_lgos_middleware(
    graph_registry: GraphRegistry,
) -> None:
    app = FastAPI()

    @app.get("/host-route")
    async def host_route() -> dict[str, bool]:
        return {"ok": True}

    LanggraphOpenaiServe(app=app, registry=graph_registry).bind_openai_api(prefix="/v1")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        host_response = await client.get("/host-route")
        lgos_response = await client.get("/v1/models")

    assert "x-request-id" not in host_response.headers
    assert lgos_response.headers.get("x-request-id")


async def test_host_handler_adds_request_fields_to_graph_node_records() -> None:
    node_logger = logging.getLogger("tests.graph_node")
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    handler.addFilter(RequestContextFilter())
    node_logger.addHandler(handler)

    def answer(_state: MessageState) -> dict:
        node_logger.warning("node.ran")
        return {"messages": [AIMessage(content="done")]}

    graph = StateGraph(MessageState).add_node("answer", answer)
    graph = graph.set_entry_point("answer").set_finish_point("answer").compile()
    registry = GraphRegistry(
        graphs={"logging": GraphConfig(graph=graph, description="Logging graph")}
    )
    app = LanggraphOpenaiServe(registry=registry).bind_openai_api(prefix="/v1").app
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "node-request"},
                json={
                    "model": "logging",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
    finally:
        node_logger.removeHandler(handler)

    assert response.status_code == status.HTTP_200_OK
    (record,) = records
    assert record.__dict__["request_id"] == "node-request"
    assert record.__dict__["model"] == "logging"


async def test_unhandled_failure_log_keeps_server_span_context() -> None:
    tracer_provider = TracerProvider()
    app = FastAPI(
        telemetry={"tracer_provider": tracer_provider, "auto_configure": False}
    )
    configure_openai_error_handlers(app)

    @app.get("/failure")
    async def fail() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    handler = TraceContextHandler()
    error_logger = logging.getLogger("langgraph_openai_serve.core.errors")
    error_logger.addHandler(handler)
    try:
        transport = ASGITransport(
            app=RequestContextMiddleware(app), raise_app_exceptions=False
        )
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/failure", headers={"X-Request-ID": "failure-request"}
            )
    finally:
        error_logger.removeHandler(handler)
        tracer_provider.shutdown()

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert [record.getMessage() for record in handler.records] == [
        "http.request.failed"
    ]
    assert handler.records[0].request_id == "failure-request"
    assert handler.contexts[0].is_valid
