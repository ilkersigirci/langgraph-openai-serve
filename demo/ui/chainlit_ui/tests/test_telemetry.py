from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from httpx2 import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


async def test_health_checks_and_socket_io_are_not_traced(
    application: ModuleType,
) -> None:
    # Global, as `opentelemetry-instrument` configures it in the container.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    transport = ASGITransport(app=application.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")
        # One Socket.IO connection carries a whole chat session.
        await client.get("/ws/socket.io/", params={"EIO": "4"})
        await client.get("/project/settings")

    servers = [
        span
        for span in exporter.get_finished_spans()
        if span.kind is trace.SpanKind.SERVER
    ]
    assert [span.attributes["url.path"] for span in servers] == ["/project/settings"]


async def test_startup_leaves_export_to_the_process_sdk(
    application: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The OTel overlay exports over gRPC through `opentelemetry-instrument`.
    # FastAPI's own environment export supports only HTTP and would duplicate it.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setattr(application, "setup_chainlit_schema", AsyncMock())
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
    sent: list[str] = []

    async def receive() -> dict[str, str]:
        return next(messages)

    async def send(message: dict[str, str]) -> None:
        sent.append(message["type"])

    await application.app(
        {"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send
    )

    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert [r.getMessage() for r in caplog.records if r.name == "fastapi"] == []
