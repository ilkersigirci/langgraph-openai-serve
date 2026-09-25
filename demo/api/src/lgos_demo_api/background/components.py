"""Hatchet background backend for the demo API."""

from hatchet_sdk import Hatchet
from langgraph_openai_serve.integrations.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_task,
)

from lgos_demo_api.core.otel import instrument_hatchet


def create_background_backend() -> HatchetBackgroundBackend:
    """Build the API-side backend that submits, reads, and cancels runs."""
    # Hatchet reads its token, namespace, and endpoints from the environment.
    hatchet = Hatchet()
    instrument_hatchet(hatchet.config)
    return HatchetBackgroundBackend(create_hatchet_task(hatchet), hatchet.runs)


__all__ = ["create_background_backend"]
