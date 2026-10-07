"""Hatchet background execution for the server and its worker process."""

import os
from collections.abc import AsyncGenerator

from hatchet_sdk import Hatchet

from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.integrations.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_task,
)
from langgraph_openai_serve.server.runtime import (
    RegistryFactory,
    open_registry,
    open_resources,
)
from langgraph_openai_serve.server.settings import ServerSettings


def create_hatchet() -> Hatchet:
    """Create a client configured by the environment and traced when exporting."""
    # The SDK reads its token, namespace, endpoints, and TLS from HATCHET_CLIENT_*.
    hatchet = Hatchet()
    if os.getenv("OTEL_TRACES_EXPORTER", "").strip().lower() not in {"", "none"}:
        from hatchet_sdk.opentelemetry.instrumentor import HatchetInstrumentor

        # opentelemetry-instrument owns export and shutdown.
        instrumentor = HatchetInstrumentor(
            config=hatchet.config, enable_hatchet_otel_collector=False
        )
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
    return hatchet


def create_hatchet_backend() -> HatchetBackgroundBackend:
    """Build the API-side backend that submits, reads, and cancels runs."""
    hatchet = create_hatchet()
    return HatchetBackgroundBackend(create_hatchet_task(hatchet), hatchet.runs)


def run_worker(factory: RegistryFactory, settings: ServerSettings) -> None:
    """Run background Responses for the same registry the API serves."""
    if settings.BACKGROUND != "hatchet":
        msg = "Set LGOS_BACKGROUND=hatchet before starting the worker."
        raise ValueError(msg)
    if (
        settings.POSTGRES_URI is not None
        and settings.HATCHET_WORKER_SLOTS >= settings.POSTGRES_POOL_SIZE
    ):
        msg = (
            "LGOS_HATCHET_WORKER_SLOTS must be below LGOS_POSTGRES_POOL_SIZE: "
            "each running interrupt holds one pool connection."
        )
        raise ValueError(msg)

    async def lifespan() -> AsyncGenerator[GraphRegistry, None]:
        async with (
            open_resources(settings) as resources,
            open_registry(factory, resources) as registry,
        ):
            yield registry

    hatchet = create_hatchet()
    hatchet.worker(
        name="lgos-worker",
        slots=settings.HATCHET_WORKER_SLOTS,
        workflows=[create_hatchet_task(hatchet)],
        lifespan=lifespan,
    ).start()
