from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lgos_api_coding_agent.app import create_app
from tests.support import message, model, openai_client, terminal


async def test_requests_are_traced_once_except_health_checks() -> None:
    # Global, as `opentelemetry-instrument` configures it in the container.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    fixture = model([message("completed", "answer", "Done", None), terminal()])
    async with openai_client(fixture) as client:
        await client.responses.create(model="coding-agent", input="Hi", store=False)
        await client.get("/health", cast_to=object)

    servers = [
        span
        for span in exporter.get_finished_spans()
        if span.kind is trace.SpanKind.SERVER
    ]
    assert [(span.name, span.attributes["http.route"]) for span in servers] == [
        ("POST /v1/responses", "/v1/responses")
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

    app = create_app(model([]))
    await app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)

    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert [
        record.getMessage() for record in caplog.records if record.name == "fastapi"
    ] == []
