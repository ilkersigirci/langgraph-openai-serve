"""Shared Hatchet background components for the demo API and agent worker."""

from hatchet_sdk import Hatchet
from langgraph_openai_serve import ResponseStore
from langgraph_openai_serve.integrations.background.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_workflows,
)


def create_hatchet_client() -> Hatchet:
    """Build a client from Hatchet's native environment configuration."""
    return Hatchet()


def create_background_backend(
    response_store: ResponseStore,
) -> HatchetBackgroundBackend:
    """Build the API-side backend that submits and cancels Hatchet runs."""
    hatchet = create_hatchet_client()
    return HatchetBackgroundBackend(
        workflow=create_hatchet_workflows(hatchet).response,
        runs=hatchet.runs,
        store=response_store,
    )


__all__ = ["create_background_backend", "create_hatchet_client"]
