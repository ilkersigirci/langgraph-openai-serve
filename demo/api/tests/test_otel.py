"""OpenTelemetry boundary tests for the demo deployment."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from lgos_demo_api import app as app_module
from lgos_demo_api.otel import instrument_fastapi_app


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
