from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI
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

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from tests.graph.support.message import make_message_graph as build_message_graph
from tests.graph.support.telemetry import Telemetry

_BASE_URL = "http://test"
_TIMEOUT = 2.0


@pytest.fixture
def anyio_backend() -> str:
    """Run the package test suite on its supported async backend."""
    return "asyncio"


@pytest.fixture
def message_graph():
    return build_message_graph()


@pytest.fixture
async def sqlite_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@pytest.fixture
def graph_registry(message_graph) -> GraphRegistry:
    return GraphRegistry(
        graphs={
            "test": GraphConfig(
                graph=message_graph,
                description="DUMMY",
            )
        }
    )


@pytest.fixture
def fastapi_app(graph_registry: GraphRegistry) -> FastAPI:
    return (
        LanggraphOpenaiServe(
            registry=graph_registry,
        )
        .bind_openai_api()
        .app
    )


@pytest.fixture
async def client(fastapi_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(
        transport=transport,
        base_url=_BASE_URL,
        timeout=_TIMEOUT,
    ) as async_client:
        yield async_client


@pytest.fixture
async def openai_client(client: AsyncClient) -> AsyncIterator[AsyncOpenAI]:
    async with AsyncOpenAI(
        api_key="test",
        base_url=f"{_BASE_URL}/v1",
        http_client=client,
        max_retries=0,
    ) as openai_client:
        yield openai_client


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
