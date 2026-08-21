"""OpenTelemetry boundary tests for the demo deployment."""

import logging
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from lgos_demo_api import app as app_module
from lgos_demo_api.otel import instrument_fastapi_app


class _TraceContextHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.contexts: list[trace.SpanContext] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.contexts.append(trace.get_current_span().get_span_context())


def test_api_instruments_the_mounted_openai_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_app = FastAPI()
    graph_serve = Mock(openai_app=openai_app)
    monkeypatch.setattr(
        app_module,
        "LanggraphOpenaiServe",
        Mock(return_value=graph_serve),
    )
    instrumented_apps: list[FastAPI] = []
    monkeypatch.setattr(app_module, "instrument_fastapi_app", instrumented_apps.append)

    host_app = app_module.create_custom_app()

    assert instrumented_apps == [openai_app]
    assert openai_app is not host_app


def test_fastapi_instrumentation_excludes_transport_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    calls: list[tuple[FastAPI, list[str]]] = []

    def instrument_app(app: FastAPI, *, exclude_spans: list[str]) -> None:
        calls.append((app, exclude_spans))

    monkeypatch.setattr(FastAPIInstrumentor, "instrument_app", instrument_app)
    app = FastAPI()

    instrument_fastapi_app(app)

    assert calls == [(app, ["send", "receive"])]


def test_fastapi_instrumentation_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    monkeypatch.delenv("OTEL_METRICS_EXPORTER", raising=False)

    app = FastAPI()
    instrument_fastapi_app(app)

    assert not getattr(app, "_is_instrumented_by_opentelemetry", False)


async def test_unhandled_failure_log_keeps_server_span_context() -> None:
    middleware_module = pytest.importorskip("langgraph_openai_serve.api.middleware")
    errors_module = pytest.importorskip("langgraph_openai_serve.core.errors")
    app = FastAPI()
    errors_module.configure_openai_error_handlers(app)

    @app.get("/failure")
    async def fail() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    tracer_provider = TracerProvider()
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        exclude_spans=["send", "receive"],
    )
    handler = _TraceContextHandler()
    error_logger = logging.getLogger("langgraph_openai_serve.core.errors")
    error_logger.addHandler(handler)

    try:
        transport = ASGITransport(
            app=middleware_module.RequestContextMiddleware(app),
            raise_app_exceptions=False,
        )
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/failure",
                headers={"X-Request-ID": "failure-request"},
            )
    finally:
        error_logger.removeHandler(handler)
        FastAPIInstrumentor.uninstrument_app(app)
        tracer_provider.shutdown()

    assert response.status_code == 500
    assert [record.getMessage() for record in handler.records] == [
        "http.request.failed"
    ]
    assert handler.records[0].request_id == "failure-request"
    assert handler.contexts[0].is_valid
