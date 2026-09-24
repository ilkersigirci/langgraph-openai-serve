"""Hatchet background backend for the demo API."""

from hatchet_sdk import Hatchet
from langgraph_openai_serve.integrations.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_task,
)


def create_background_backend() -> HatchetBackgroundBackend:
    """Build the API-side backend that submits, reads, and cancels runs."""
    # Hatchet reads its token, namespace, and endpoints from the environment.
    hatchet = Hatchet()
    return HatchetBackgroundBackend(create_hatchet_task(hatchet), hatchet.runs)


__all__ = ["create_background_backend"]
