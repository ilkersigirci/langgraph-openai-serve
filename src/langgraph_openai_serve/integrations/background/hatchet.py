"""Optional Hatchet integration for durable background Responses."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated

from hatchet_sdk import ConcurrencyExpression, ConcurrencyLimitStrategy
from hatchet_sdk.clients.rest.models.v1_task_status import V1TaskStatus
from hatchet_sdk.context.context import (
    Context,  # ruff: ignore[typing-only-third-party-import] - Hatchet resolves annotations while registering.
)
from hatchet_sdk.exceptions import IdempotencyCollisionError
from hatchet_sdk.features.runs import BulkCancelReplayOpts, RunFilter
from hatchet_sdk.runnables.types import EmptyModel
from hatchet_sdk.runnables.workflow import Standalone, Workflow
from hatchet_sdk.types.idempotency import TTLBasedIdempotencyConfig
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from langgraph_openai_serve.background.contracts import BackgroundSettings
from langgraph_openai_serve.background.worker import BackgroundWorker

if TYPE_CHECKING:
    from hatchet_sdk import Hatchet
    from hatchet_sdk.features.runs import RunsClient

    from langgraph_openai_serve.background.store import ResponseStore, StoredRun

# Hatchet run metadata that lets cancellation find a run by Response ID.
_RESPONSE_ID_METADATA = "lgos_response_id"
# Hatchet run metadata that keys concurrency by the shared checkpoint thread.
_CHECKPOINT_THREAD_METADATA = "lgos_checkpoint_thread_id"


class HatchetAdapterSettings(BaseModel):
    """Native Hatchet scheduling, timeout, retry, and cron settings."""

    workflow_name: str = "lgos-background-response"
    maintenance_task_name: str = "lgos-background-maintenance"
    retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=2.0, gt=0)
    backoff_max_seconds: int = Field(default=30, ge=1)
    schedule_timeout: timedelta = Field(
        default=timedelta(minutes=30),
        gt=timedelta(0),
    )
    execution_timeout: timedelta = Field(
        default=timedelta(minutes=20),
        gt=timedelta(0),
    )
    finalization_retries: int = Field(default=3, ge=0)
    finalization_timeout: timedelta = Field(
        default=timedelta(minutes=5),
        gt=timedelta(0),
    )
    maintenance_cron: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ] = "*/5 * * * *"
    maintenance_timeout: timedelta = Field(
        default=timedelta(minutes=5),
        gt=timedelta(0),
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def max_queue_time(self) -> timedelta:
        """
        Longest time Hatchet can keep a submitted run queued.

        Hatchet retries a schedule timeout like any other failure, so each
        attempt may wait ``schedule_timeout`` plus its backoff.
        """
        attempt = self.schedule_timeout + timedelta(seconds=self.backoff_max_seconds)
        return attempt * (self.retries + 1)


class HatchetResponseInput(BaseModel):
    """Workflow input containing only a persisted Response reference."""

    response_id: str

    model_config = ConfigDict(extra="forbid", frozen=True)


HatchetResponseWorkflow = Workflow[HatchetResponseInput]
HatchetMaintenanceTask = Standalone[EmptyModel, dict[str, int]]


@dataclass(frozen=True, slots=True)
class HatchetWorkflows:
    """Hatchet registrations used by both the API and worker processes."""

    response: HatchetResponseWorkflow
    maintenance: HatchetMaintenanceTask

    @property
    def registrations(self) -> tuple[HatchetResponseWorkflow, HatchetMaintenanceTask]:
        """Native objects passed directly to ``hatchet.worker``."""
        return (self.response, self.maintenance)


class HatchetBackgroundBackend:
    """Run background Responses as Hatchet workflows."""

    def __init__(
        self,
        *,
        workflow: HatchetResponseWorkflow,
        runs: RunsClient,
        store: ResponseStore,
        settings: BackgroundSettings | None = None,
    ) -> None:
        self._workflow = workflow
        self._runs = runs
        self.store = store
        self.settings = settings or BackgroundSettings()

    async def submit(self, run: StoredRun) -> None:
        """Trigger the workflow without waiting for it."""
        await _trigger(self._workflow, run)

    async def stop(self, run: StoredRun) -> None:
        """Cancel the Hatchet run found by its Response ID metadata."""
        # A run this misses finds its Response terminal and only deletes
        # checkpoints, so eventual consistency in Hatchet's run listing is safe.
        await self._runs.aio_bulk_cancel(
            BulkCancelReplayOpts(
                filters=RunFilter(
                    since=run.created_at,
                    statuses=[V1TaskStatus.QUEUED, V1TaskStatus.RUNNING],
                    additional_metadata={_RESPONSE_ID_METADATA: run.response_id},
                )
            )
        )


def create_hatchet_workflows(
    hatchet: Hatchet,
    *,
    settings: HatchetAdapterSettings | None = None,
) -> HatchetWorkflows:
    """
    Register execution, final failure, and maintenance directly in Hatchet.

    Tasks run the ``BackgroundWorker`` yielded by the Hatchet worker lifespan,
    so database resources can open inside that lifespan.
    """
    task_settings = settings or HatchetAdapterSettings()
    workflow = hatchet.workflow(
        name=task_settings.workflow_name,
        input_validator=HatchetResponseInput,
        # Resubmitting a queued run is then a no-op while Hatchet can still
        # hold the first submission; after that, Hatchet has lost it.
        idempotency=TTLBasedIdempotencyConfig(
            key_expression="input.response_id",
            ttl=task_settings.max_queue_time,
        ),
        # An interrupt answer is a new Response on its paused run's checkpoint
        # thread; queue it behind that thread's active run instead of failing
        # on the busy coordinator lease.
        concurrency=ConcurrencyExpression(
            expression=f"additional_metadata.{_CHECKPOINT_THREAD_METADATA}",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @workflow.task(
        name=f"{task_settings.workflow_name}-execute",
        retries=task_settings.retries,
        backoff_factor=task_settings.backoff_factor,
        backoff_max_seconds=task_settings.backoff_max_seconds,
        schedule_timeout=task_settings.schedule_timeout,
        execution_timeout=task_settings.execution_timeout,
    )
    async def execute(
        job_input: HatchetResponseInput,
        context: Context,
    ) -> dict[str, str]:
        await _lifespan_worker(context).execute(job_input.response_id)
        return {"response_id": job_input.response_id}

    @workflow.on_failure_task(
        name=f"{task_settings.workflow_name}-finalize",
        retries=task_settings.finalization_retries,
        backoff_factor=task_settings.backoff_factor,
        backoff_max_seconds=task_settings.backoff_max_seconds,
        schedule_timeout=task_settings.schedule_timeout,
        execution_timeout=task_settings.finalization_timeout,
    )
    async def finalize(
        job_input: HatchetResponseInput,
        context: Context,
    ) -> dict[str, str]:
        await _lifespan_worker(context).finalize(job_input.response_id)
        return {"response_id": job_input.response_id}

    @hatchet.task(
        name=task_settings.maintenance_task_name,
        on_crons=[task_settings.maintenance_cron],
        concurrency=1,
        retries=task_settings.finalization_retries,
        backoff_factor=task_settings.backoff_factor,
        backoff_max_seconds=task_settings.backoff_max_seconds,
        schedule_timeout=task_settings.schedule_timeout,
        execution_timeout=task_settings.maintenance_timeout,
    )
    async def maintain(
        _input: EmptyModel,
        context: Context,
    ) -> dict[str, int]:
        return await _lifespan_worker(context).maintain(
            resubmit=lambda run: _trigger(workflow, run)
        )

    return HatchetWorkflows(response=workflow, maintenance=maintain)


async def _trigger(workflow: HatchetResponseWorkflow, run: StoredRun) -> None:
    # A collision means this Response was already submitted.
    with suppress(IdempotencyCollisionError):
        await workflow.aio_run(
            HatchetResponseInput(response_id=run.response_id),
            wait_for_result=False,
            additional_metadata={
                _RESPONSE_ID_METADATA: run.response_id,
                _CHECKPOINT_THREAD_METADATA: run.checkpoint_thread_id,
            },
        )


def _lifespan_worker(context: Context) -> BackgroundWorker:
    worker = context.lifespan
    if not isinstance(worker, BackgroundWorker):
        msg = "The Hatchet worker lifespan must yield a BackgroundWorker."
        raise TypeError(msg)
    return worker


__all__ = [
    "HatchetAdapterSettings",
    "HatchetBackgroundBackend",
    "HatchetMaintenanceTask",
    "HatchetResponseInput",
    "HatchetResponseWorkflow",
    "HatchetWorkflows",
    "create_hatchet_workflows",
]
