"""Shared Hatchet background components for the demo API and agent worker."""

from dataclasses import dataclass
from datetime import timedelta

from hatchet_sdk import Hatchet
from hatchet_sdk.config import ClientConfig
from langgraph_openai_serve import (
    BackgroundSettings,
    BackgroundWorker,
    GraphRegistry,
)
from langgraph_openai_serve.integrations.background_postgres import (
    PostgresResponseStore,
)
from langgraph_openai_serve.integrations.hatchet import (
    HatchetAdapterSettings,
    HatchetBackgroundBackend,
    HatchetWorkflows,
    create_hatchet_workflows,
)

from lgos_demo_api.settings import settings


@dataclass(frozen=True, slots=True)
class BackgroundComponents:
    """Process-local Hatchet client, workflows, worker, and API backend."""

    hatchet: Hatchet
    workflows: HatchetWorkflows
    worker: BackgroundWorker
    backend: HatchetBackgroundBackend


def create_background_settings() -> BackgroundSettings:
    """Resolve the demo's bounded core execution and retention settings."""
    return BackgroundSettings(
        admission_capacity=settings.BACKGROUND_ADMISSION_CAPACITY,
        result_retention=timedelta(
            seconds=settings.BACKGROUND_RESULT_RETENTION_SECONDS
        ),
        stored_result_retention=timedelta(
            seconds=settings.BACKGROUND_STORED_RESULT_RETENTION_SECONDS
        ),
        idempotency_retention=timedelta(
            seconds=settings.BACKGROUND_IDEMPOTENCY_RETENTION_SECONDS
        ),
    )


def create_hatchet_settings() -> HatchetAdapterSettings:
    """Resolve native Hatchet workflow retry, timeout, and deduplication settings."""
    return HatchetAdapterSettings(
        workflow_name=settings.HATCHET_TASK_NAME,
        retries=settings.HATCHET_TASK_RETRIES,
        schedule_timeout=timedelta(seconds=settings.HATCHET_SCHEDULE_TIMEOUT_SECONDS),
        execution_timeout=timedelta(seconds=settings.HATCHET_EXECUTION_TIMEOUT_SECONDS),
        idempotency_ttl=timedelta(seconds=settings.HATCHET_IDEMPOTENCY_TTL_SECONDS),
    )


def create_hatchet_client() -> Hatchet:
    """Build a client from Hatchet's native environment configuration."""
    return Hatchet(config=ClientConfig(namespace=settings.HATCHET_NAMESPACE))


def create_background_components(
    graphs: GraphRegistry,
    response_store: PostgresResponseStore,
) -> BackgroundComponents:
    """Build equivalent components in either the API or worker process."""
    background_settings = create_background_settings()
    adapter_settings = create_hatchet_settings()
    hatchet = create_hatchet_client()
    worker = BackgroundWorker(
        graphs=graphs,
        store=response_store,
        settings=background_settings,
    )
    workflows = create_hatchet_workflows(hatchet, worker, settings=adapter_settings)
    backend = HatchetBackgroundBackend(
        workflow=workflows.response,
        runs=hatchet.runs,
        store=response_store,
        settings=background_settings,
    )
    return BackgroundComponents(
        hatchet=hatchet,
        workflows=workflows,
        worker=worker,
        backend=backend,
    )


__all__ = [
    "BackgroundComponents",
    "create_background_components",
    "create_background_settings",
    "create_hatchet_client",
    "create_hatchet_settings",
]
