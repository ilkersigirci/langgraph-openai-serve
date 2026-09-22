"""Optional OpenTelemetry setup for the demo API."""

from __future__ import annotations

import os

from fastapi import FastAPI


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
