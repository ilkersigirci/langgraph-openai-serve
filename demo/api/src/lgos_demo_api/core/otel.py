"""Optional OpenTelemetry setup for the demo API and background worker."""

from __future__ import annotations

import os

from fastapi import FastAPI
from hatchet_sdk import ClientConfig


def instrument_hatchet(config: ClientConfig) -> None:
    """Use the deployment's tracer provider for Hatchet submission and execution."""
    if os.getenv("OTEL_TRACES_EXPORTER", "").strip().lower() in {"", "none"}:
        return

    from hatchet_sdk.opentelemetry.instrumentor import HatchetInstrumentor

    # opentelemetry-instrument owns export and shutdown through the local Collector.
    instrumentor = HatchetInstrumentor(
        config=config,
        enable_hatchet_otel_collector=False,
    )
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()


def instrument_fastapi_app(app: FastAPI) -> None:
    """Instrument the mounted API when the deployment exports OTel signals."""
    if not _otel_signal_export_enabled():
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app,
        exclude_spans=["send", "receive"],
    )


def _otel_signal_export_enabled() -> bool:
    return any(
        os.getenv(name, "").strip().lower() not in {"", "none"}
        for name in ("OTEL_TRACES_EXPORTER", "OTEL_METRICS_EXPORTER")
    )
