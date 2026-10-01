import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tests.graph.support.telemetry import Telemetry


@pytest.fixture(scope="session")
def telemetry_sdk() -> Telemetry:
    """Install the SDK once; OpenTelemetry's global providers cannot be replaced."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    # Delta temporality makes every collection return only new measurements.
    metric_reader = InMemoryMetricReader(
        preferred_temporality={Histogram: AggregationTemporality.DELTA}
    )
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
    return Telemetry(span_exporter, metric_reader)


@pytest.fixture
def telemetry(telemetry_sdk: Telemetry) -> Telemetry:
    """Return the SDK without the signals of earlier tests."""
    telemetry_sdk.span_exporter.clear()
    telemetry_sdk.metric_reader.get_metrics_data()
    return telemetry_sdk
