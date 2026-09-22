"""Shared Hatchet background components for the demo API and agent worker."""

from dataclasses import dataclass

from hatchet_sdk import Hatchet
from langgraph_openai_serve import (
    BackgroundSettings,
    BackgroundWorker,
    GraphRegistry,
    ResponseStore,
)
from langgraph_openai_serve.integrations.background.hatchet import (
    HatchetBackgroundBackend,
    HatchetWorkflows,
    create_hatchet_workflows,
)

from lgos_demo_api.core.settings import settings


@dataclass(frozen=True, slots=True)
class BackgroundComponents:
    """Process-local Hatchet client, workflows, worker, and API backend."""

    hatchet: Hatchet
    workflows: HatchetWorkflows
    worker: BackgroundWorker
    backend: HatchetBackgroundBackend


def create_background_settings() -> BackgroundSettings:
    """Resolve the demo's global background admission bound."""
    return BackgroundSettings(
        admission_capacity=settings.BACKGROUND_ADMISSION_CAPACITY,
    )


def create_hatchet_client() -> Hatchet:
    """Build a client from Hatchet's native environment configuration."""
    return Hatchet()


def create_background_components(
    graphs: GraphRegistry,
    response_store: ResponseStore,
) -> BackgroundComponents:
    """Build equivalent components in either the API or worker process."""
    background_settings = create_background_settings()
    hatchet = create_hatchet_client()
    worker = BackgroundWorker(
        graphs=graphs,
        store=response_store,
        settings=background_settings,
    )
    workflows = create_hatchet_workflows(hatchet, worker)
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
]
