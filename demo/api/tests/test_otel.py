"""OpenTelemetry boundary tests for the demo deployment."""

import logging
from logging.config import DictConfigurator
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from hatchet_sdk import ClientConfig, Hatchet
from hatchet_sdk.opentelemetry import instrumentor as hatchet_otel
from httpx2 import ASGITransport, AsyncClient, MockTransport, Request, Response
from openai import AsyncOpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPX2ClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from lgos_demo_api import app as app_module
from lgos_demo_api.background import components as background_components
from lgos_demo_api.background import worker as background_worker
from lgos_demo_api.core.logging import LOGGING_CONFIG
from lgos_demo_api.core.otel import instrument_fastapi_app, instrument_hatchet


@pytest.mark.parametrize("exporter_setting", [None, "none", " NONE "])
def test_hatchet_instrumentation_requires_trace_export(
    monkeypatch: pytest.MonkeyPatch,
    exporter_setting: str | None,
) -> None:
    if exporter_setting is None:
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    else:
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", exporter_setting)
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    monkeypatch.setattr(
        hatchet_otel,
        "HatchetInstrumentor",
        Mock(side_effect=AssertionError("Tracing is disabled")),
    )

    instrument_hatchet(Mock(spec=ClientConfig))


@pytest.mark.parametrize("entrypoint", ["api", "worker"])
def test_hatchet_entrypoints_enable_native_tracing_once(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    config = Mock(spec=ClientConfig)
    hatchet = Mock(spec=Hatchet, config=config)
    instrumentor = Mock(is_instrumented_by_opentelemetry=False)

    def mark_instrumented() -> None:
        instrumentor.is_instrumented_by_opentelemetry = True

    instrumentor.instrument.side_effect = mark_instrumented
    instrumentor_factory = Mock(return_value=instrumentor)
    monkeypatch.setattr(hatchet_otel, "HatchetInstrumentor", instrumentor_factory)
    logging_setup = Mock()

    if entrypoint == "api":
        monkeypatch.setattr(background_components, "Hatchet", lambda: hatchet)
        start = background_components.create_background_backend
    else:
        monkeypatch.setattr(background_worker, "Hatchet", lambda: hatchet)
        monkeypatch.setattr(background_worker.settings, "BACKGROUND_ENABLED", True)
        monkeypatch.setattr(background_worker, "configure_logging", logging_setup)
        start = background_worker.main

    start()
    start()

    instrumentor_factory.assert_called_with(
        config=config,
        enable_hatchet_otel_collector=False,
    )
    instrumentor.instrument.assert_called_once_with()
    if entrypoint == "worker":
        logging_setup.assert_called_with(root_level=logging.INFO)


class _TraceContextHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.contexts: list[trace.SpanContext] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.contexts.append(trace.get_current_span().get_span_context())


def test_hatchet_logs_reach_root_handlers_with_trace_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("hatchet")
    monkeypatch.setattr(logger, "handlers", logger.handlers.copy())
    for attribute in ("level", "propagate", "disabled"):
        monkeypatch.setattr(logger, attribute, getattr(logger, attribute))
    # Apply only this logger's configuration without replacing pytest's handlers.
    DictConfigurator(LOGGING_CONFIG).configure_logger(
        "hatchet", LOGGING_CONFIG["loggers"]["hatchet"]
    )
    handler = _TraceContextHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    provider = TracerProvider()
    try:
        with provider.get_tracer(__name__).start_as_current_span("task") as span:
            logger.info("task.started")
    finally:
        root.removeHandler(handler)
        provider.shutdown()

    assert [record.getMessage() for record in handler.records] == ["task.started"]
    assert handler.contexts == [span.get_span_context()]
    assert not logger.handlers


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


async def test_httpx2_instrumentation_supports_openai_v3() -> None:
    async def respond(_request: Request) -> Response:
        return Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "instrumented-model",
                        "object": "model",
                        "created": 0,
                        "owned_by": "test",
                    }
                ],
            },
        )

    http_client = AsyncClient(transport=MockTransport(respond))
    HTTPX2ClientInstrumentor.instrument_client(http_client)
    try:
        async with AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
        ) as client:
            models = await client.models.list()
    finally:
        HTTPX2ClientInstrumentor.uninstrument_client(http_client)

    assert models.data[0].id == "instrumented-model"


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
