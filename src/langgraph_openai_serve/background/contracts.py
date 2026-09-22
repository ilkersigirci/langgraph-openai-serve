"""Public contracts for polling-only background Responses execution."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

if TYPE_CHECKING:
    from langgraph_openai_serve.background.store import NewRun, StoredRun


class BackgroundPolicy(BaseModel):
    """Opt one registered model into versioned background execution."""

    version: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    model_config = ConfigDict(extra="forbid", frozen=True)


class BackgroundSettings(BaseModel):
    """LGOS-owned admission and Response-retention settings."""

    admission_capacity: int = Field(default=10_000, ge=1)
    result_retention: timedelta = Field(
        default=timedelta(hours=1),
        gt=timedelta(0),
    )
    stored_result_retention: timedelta = Field(
        default=timedelta(days=30),
        gt=timedelta(0),
    )
    idempotency_retention: timedelta = Field(
        default=timedelta(hours=24),
        gt=timedelta(0),
    )
    maintenance_batch_size: int = Field(default=100, ge=1, le=10_000)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def result_retention_for(self, *, stored: bool) -> timedelta:
        """Return the configured public-result retention for one Response."""
        return self.stored_result_retention if stored else self.result_retention


@runtime_checkable
class BackgroundBackend(Protocol):
    """
    Own the complete lifecycle behind the OpenAI polling API.

    Hatchet is LGOS's supplied durable implementation. Applications may implement
    this boundary to own scheduling, cancellation, and recovery. Replacing only
    persistence needs a ResponseStore implementation, not a new backend.
    Backend construction and resource lifetime belong to the application.
    """

    async def create(self, run: NewRun) -> StoredRun:
        """Persist and submit a validated background Response."""
        ...

    async def retrieve(self, response_id: str, owner_scope: str) -> StoredRun | None:
        """Read one authorized, unexpired public Response snapshot."""
        ...

    async def cancel(
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        stored: bool,
    ) -> StoredRun | None:
        """Cancel using the persisted Response's normalized storage decision."""
        ...


class RetryableJobError(RuntimeError):
    """Signal an infrastructure failure that the background engine should retry."""


__all__ = [
    "BackgroundBackend",
    "BackgroundPolicy",
    "BackgroundSettings",
    "RetryableJobError",
]
