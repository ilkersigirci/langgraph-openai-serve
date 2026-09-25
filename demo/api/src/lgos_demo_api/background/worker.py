"""Independent Hatchet worker entry point for background Responses."""

import logging
from collections.abc import AsyncGenerator

from hatchet_sdk import Hatchet
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.integrations.hatchet import create_hatchet_task

from lgos_demo_api.core.logging import configure_logging
from lgos_demo_api.core.otel import instrument_hatchet
from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs.advanced_graph import (
    create_advanced_graph_config,
    open_advanced_graph,
)
from lgos_demo_api.graphs.background_mock import background_mock_graph_config
from lgos_demo_api.persistence.postgres import postgres_runtime


async def _lifespan() -> AsyncGenerator[GraphRegistry, None]:
    """Yield the graphs that Hatchet tasks read from ``context.lifespan``."""
    async with (
        postgres_runtime(settings.POSTGRES_URI) as runtime,
        open_advanced_graph(runtime.checkpointer, runtime.store) as advanced_graph,
    ):
        # Register every background-capable model under its API model ID.
        yield GraphRegistry(
            registry={
                "advanced-graph": create_advanced_graph_config(
                    lambda: advanced_graph,
                    runtime.run_coordinator,
                ),
                "background-mock": background_mock_graph_config,
            }
        )


def main() -> None:
    """Register and run the bounded-capacity background agent worker."""
    if not settings.BACKGROUND_ENABLED:
        msg = "Set DEMO_API_BACKGROUND_ENABLED=True before starting the worker."
        raise RuntimeError(msg)

    # Hatchet uses the root logger's level for its task-log forwarding handler.
    configure_logging(root_level=logging.INFO)
    hatchet = Hatchet()
    instrument_hatchet(hatchet.config)
    worker = hatchet.worker(
        name="background-agent-worker",
        slots=settings.HATCHET_WORKER_SLOTS,
        workflows=[create_hatchet_task(hatchet)],
        lifespan=_lifespan,
    )
    worker.start()


if __name__ == "__main__":
    main()
