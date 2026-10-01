from unittest.mock import Mock

from httpx2 import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lgos_files_api import FileRepository, create_files_app
from lgos_files_api.contracts import FilePage


async def test_requests_are_traced_once_except_health_checks() -> None:
    # Global, as `opentelemetry-instrument` configures it in the container.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    repository = Mock(spec=FileRepository)
    repository.list_files.return_value = FilePage(data=[], has_more=False)

    transport = ASGITransport(app=create_files_app(repository))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/v1/files")
        await client.get("/health")

    servers = [
        span
        for span in exporter.get_finished_spans()
        if span.kind is trace.SpanKind.SERVER
    ]
    assert [(span.name, span.attributes["http.route"]) for span in servers] == [
        ("GET /v1/files", "/v1/files")
    ]


async def test_startup_leaves_export_to_the_process_sdk(monkeypatch, caplog) -> None:
    # The OTel overlay exports over gRPC through `opentelemetry-instrument`.
    # FastAPI's own environment export supports only HTTP and would duplicate it.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
    sent: list[str] = []

    async def receive():
        return next(messages)

    async def send(message) -> None:
        sent.append(message["type"])

    app = create_files_app(Mock(spec=FileRepository))
    await app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)

    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert [
        record.getMessage() for record in caplog.records if record.name == "fastapi"
    ] == []
