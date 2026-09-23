"""Public contracts for polling-only background Responses execution."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from openai.types.responses import Response

    from langgraph_openai_serve.background.store import ResponseStore, StoredRun


class BackgroundSettings(BaseModel):
    """LGOS-owned Response retention and maintenance settings."""

    # Defaults follow OpenAI: store=false background Responses stay retrievable
    # for roughly 10 minutes, stored ones for 30 days.
    result_retention: timedelta = Field(
        default=timedelta(minutes=10),
        gt=timedelta(0),
    )
    stored_result_retention: timedelta = Field(
        default=timedelta(days=30),
        gt=timedelta(0),
    )
    # A run still queued this long may never have reached the engine, so
    # maintenance submits it again. Engines must ignore duplicate submits.
    resubmit_after: timedelta = Field(
        default=timedelta(minutes=1),
        gt=timedelta(0),
    )
    maintenance_batch_size: int = Field(default=100, ge=1, le=10_000)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def result_retention_for(self, response: Response) -> timedelta:
        """Return the retention chosen by the Response's ``store`` flag."""
        # The OpenAI SDK type omits ``store``; LGOS serializes it as an extra.
        stored = (response.model_extra or {}).get("store") is True
        return self.stored_result_retention if stored else self.result_retention


class BackgroundBackend(Protocol):
    """
    Start and stop execution behind the OpenAI polling API.

    LGOS owns every Response state change in ``store``; a backend only runs
    work. Hatchet is LGOS's supplied durable implementation. Replacing only
    persistence needs a ResponseStore implementation, not a new backend.
    Backend construction and resource lifetime belong to the application.
    """

    store: ResponseStore
    settings: BackgroundSettings

    async def submit(self, run: StoredRun) -> None:
        """
        Start executing a persisted run.

        Maintenance also calls this for runs that stayed queued, so submitting
        an already submitted run must not start a second execution.
        """
        ...

    async def stop(self, run: StoredRun) -> None:
        """Stop the work of a run whose cancellation already won."""
        ...


__all__ = [
    "BackgroundBackend",
    "BackgroundSettings",
]
