"""Independent Hatchet worker entry point for the background report agent."""

from collections.abc import AsyncGenerator

from langgraph_openai_serve import BackgroundWorker, GraphRegistry
from langgraph_openai_serve.integrations.background.hatchet import (
    create_hatchet_workflows,
)

from lgos_demo_api.background.components import create_hatchet_client
from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs.background_report import (
    create_background_report_config,
    create_background_report_graph,
)
from lgos_demo_api.persistence.postgres import postgres_runtime


async def _lifespan() -> AsyncGenerator[BackgroundWorker, None]:
    """Yield the worker that Hatchet tasks read from ``context.lifespan``."""
    async with postgres_runtime(settings.POSTGRES_URI) as runtime:
        graph = create_background_report_graph(runtime.checkpointer)
        registry = GraphRegistry(
            registry={
                "background-report-agent": create_background_report_config(
                    lambda: graph,
                    runtime.run_coordinator,
                )
            }
        )
        yield BackgroundWorker(graphs=registry, store=runtime.response_store)


def main() -> None:
    """Register and run the bounded-capacity background agent worker."""
    if not settings.BACKGROUND_ENABLED:
        msg = "Set DEMO_API_BACKGROUND_ENABLED=True before starting the worker."
        raise RuntimeError(msg)

    hatchet = create_hatchet_client()
    workflows = create_hatchet_workflows(hatchet)
    native_worker = hatchet.worker(
        name="background-agent-worker",
        slots=settings.HATCHET_WORKER_SLOTS,
        workflows=list(workflows.registrations),
        lifespan=_lifespan,
    )
    native_worker.start()


if __name__ == "__main__":
    main()
