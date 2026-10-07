"""OpenTelemetry boundary of the ``lgos serve`` and ``lgos worker`` processes."""

import logging
from logging.config import DictConfigurator
from unittest.mock import Mock

import pytest
from hatchet_sdk import ClientConfig, Hatchet
from hatchet_sdk.opentelemetry import instrumentor as hatchet_otel
from httpx2 import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from langgraph_openai_serve.server import create_app, hatchet as server_hatchet
from langgraph_openai_serve.server.logging import logging_config
from tests.graph.support.telemetry import Telemetry, TraceContextHandler
from tests.server.support import create_registry, server_settings


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
    monkeypatch.setattr(server_hatchet, "Hatchet", Mock(spec=Hatchet))
    monkeypatch.setattr(
        hatchet_otel,
        "HatchetInstrumentor",
        Mock(side_effect=AssertionError("Tracing is disabled")),
    )

    server_hatchet.create_hatchet()


def test_hatchet_clients_enable_native_tracing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    config = Mock(spec=ClientConfig)
    monkeypatch.setattr(
        server_hatchet, "Hatchet", lambda: Mock(spec=Hatchet, config=config)
    )
    instrumentor = Mock(is_instrumented_by_opentelemetry=False)

    def mark_instrumented() -> None:
        instrumentor.is_instrumented_by_opentelemetry = True

    instrumentor.instrument.side_effect = mark_instrumented
    instrumentor_factory = Mock(return_value=instrumentor)
    monkeypatch.setattr(hatchet_otel, "HatchetInstrumentor", instrumentor_factory)

    # The API backend and the worker each create a client.
    server_hatchet.create_hatchet()
    server_hatchet.create_hatchet()

    instrumentor_factory.assert_called_with(
        config=config, enable_hatchet_otel_collector=False
    )
    instrumentor.instrument.assert_called_once_with()


def test_hatchet_logs_reach_root_handlers_with_trace_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("hatchet")
    monkeypatch.setattr(logger, "handlers", logger.handlers.copy())
    for attribute in ("level", "propagate", "disabled"):
        monkeypatch.setattr(logger, attribute, getattr(logger, attribute))
    config = logging_config("my_app", root_level=logging.INFO)
    # Apply only this logger's configuration without replacing pytest's handlers.
    DictConfigurator(config).configure_logger("hatchet", config["loggers"]["hatchet"])
    handler = TraceContextHandler()
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


async def test_requests_are_traced_once_except_health_checks(
    telemetry: Telemetry,
) -> None:
    # The global provider stands in for the one `opentelemetry-instrument` installs.
    app = create_app(create_registry, settings=server_settings())
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        await http.get("/v1/models")
        await http.get("/v1/health")

    servers = [
        span
        for span in telemetry.span_exporter.get_finished_spans()
        if span.kind is trace.SpanKind.SERVER
    ]
    assert [(span.name, span.attributes["http.route"]) for span in servers] == [
        ("GET /v1/models", "/v1/models")
    ]


async def test_startup_leaves_export_to_the_process_sdk(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # `opentelemetry-instrument` exports over gRPC; FastAPI's own environment
    # export supports only HTTP and would duplicate it.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    app = create_app(create_registry, settings=server_settings())

    async with app.router.lifespan_context(app):
        pass

    assert [r.getMessage() for r in caplog.records if r.name == "fastapi"] == []
