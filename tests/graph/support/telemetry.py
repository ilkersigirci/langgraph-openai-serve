from dataclasses import dataclass

from opentelemetry.sdk.metrics.export import HistogramDataPoint, InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@dataclass(frozen=True)
class Telemetry:
    """The in-memory SDK signals of the graph runs a test executes."""

    span_exporter: InMemorySpanExporter
    metric_reader: InMemoryMetricReader

    def workflow_spans(self) -> list[ReadableSpan]:
        return [
            span
            for span in self.span_exporter.get_finished_spans()
            if span.name.startswith("invoke_workflow")
        ]

    def workflow_durations(self) -> list[HistogramDataPoint]:
        data = self.metric_reader.get_metrics_data()
        if data is None:
            return []
        return [
            point
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            if metric.name == "gen_ai.invoke_workflow.duration"
            for point in metric.data.data_points
            if isinstance(point, HistogramDataPoint)
        ]
