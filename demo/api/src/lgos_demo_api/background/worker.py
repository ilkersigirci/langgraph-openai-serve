"""Independent Hatchet worker entry point for the background report agent."""

from collections.abc import AsyncGenerator

from langgraph_openai_serve import (
    BackgroundSettings,
    BackgroundWorker,
    GraphRegistry,
    ResponseStore,
    RetryableJobError,
)
from langgraph_openai_serve.integrations.hatchet import create_hatchet_workflows

from lgos_demo_api.background.components import (
    create_background_settings,
    create_hatchet_client,
)
from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs.background_report import (
    create_background_report_config,
    create_background_report_graph,
)
from lgos_demo_api.persistence.postgres import postgres_runtime


class _LifespanWorker:
    """Delay database-backed worker construction until Hatchet opens lifespan."""

    def __init__(self, worker_settings: BackgroundSettings) -> None:
        self.settings = worker_settings
        self.worker: BackgroundWorker | None = None

    def _require_worker(self) -> BackgroundWorker:
        if self.worker is None:
            msg = "The background worker lifespan is not ready."
            raise RetryableJobError(msg)
        return self.worker

    @property
    def store(self) -> ResponseStore:
        """Expose the ready store to Hatchet's maintenance delivery pass."""
        return self._require_worker().store

    async def execute(self, response_id: str) -> None:
        await self._require_worker().execute(response_id)

    async def finalize(self, response_id: str) -> None:
        await self._require_worker().finalize(response_id)

    async def maintain(self) -> dict[str, int]:
        return await self._require_worker().maintain()


def main() -> None:
    """Register and run the bounded-capacity background agent worker."""
    if not settings.BACKGROUND_ENABLED:
        msg = "Set DEMO_API_BACKGROUND_ENABLED=True before starting the worker."
        raise RuntimeError(msg)

    executor = _LifespanWorker(create_background_settings())
    hatchet = create_hatchet_client()
    workflows = create_hatchet_workflows(hatchet, executor)

    async def lifespan() -> AsyncGenerator[None, None]:
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
            executor.worker = BackgroundWorker(
                graphs=registry,
                store=runtime.response_store,
                settings=executor.settings,
            )
            try:
                yield
            finally:
                executor.worker = None

    native_worker = hatchet.worker(
        name="background-agent-worker",
        slots=settings.HATCHET_WORKER_SLOTS,
        workflows=list(workflows.registrations),
        lifespan=lifespan,
    )
    native_worker.start()


if __name__ == "__main__":
    main()
