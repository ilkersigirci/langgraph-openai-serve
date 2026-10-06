"""Optional Hatchet tracing for the demo API and background worker."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
