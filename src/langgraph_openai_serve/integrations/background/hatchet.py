"""Optional Hatchet integration for durable background Responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Protocol

from anyio import CancelScope
from hatchet_sdk.context.context import (
    Context,  # ruff: ignore[typing-only-third-party-import] - Hatchet resolves annotations while registering.
)
from hatchet_sdk.exceptions import IdempotencyCollisionError
from hatchet_sdk.runnables.types import EmptyModel
from hatchet_sdk.runnables.workflow import Standalone, Workflow
from hatchet_sdk.types.idempotency import TTLBasedIdempotencyConfig
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from langgraph_openai_serve.background.contracts import BackgroundSettings
from langgraph_openai_serve.background.store import (
    NewRun,
    ResponseStatus,
    ResponseStore,
    StoredRun,
)
from langgraph_openai_serve.core.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from hatchet_sdk import Hatchet
    from hatchet_sdk.features.runs import RunsClient


class BackgroundJobExecutor(Protocol):
    """Core work invoked by the registered Hatchet workflows."""

    @property
    def store(self) -> ResponseStore:
        """Response store shared with Hatchet maintenance."""
        ...

    @property
    def settings(self) -> BackgroundSettings:
        """LGOS-owned maintenance limits."""
        ...

    async def execute(self, response_id: str) -> None:
        """Execute or resume one persisted run."""
        ...

    async def finalize(self, response_id: str) -> None:
        """Publish a result after native retries are exhausted."""
        ...

    async def maintain(self) -> dict[str, int]:
        """Clean checkpoints and expire retained Response data."""
        ...


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
    idempotency_ttl: timedelta = Field(
        default=timedelta(hours=24),
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
    """Persist OpenAI state and delegate the complete run lifecycle to Hatchet."""

    def __init__(
        self,
        *,
        workflow: HatchetResponseWorkflow,
        runs: RunsClient,
        store: ResponseStore,
        settings: BackgroundSettings | None = None,
    ) -> None:
        if not isinstance(store, ResponseStore):
            msg = "store must implement ResponseStore."
            raise TypeError(msg)
        self._workflow = workflow
        self._runs = runs
        self.store = store
        self.settings = settings or BackgroundSettings()

    async def create(self, run: NewRun) -> StoredRun:
        """Persist the run, then best-effort submit it to Hatchet."""
        accepted = await self.store.accept(
            run,
            capacity=self.settings.admission_capacity,
        )
        if accepted.terminal or accepted.workflow_run_id is not None:
            return accepted

        # A failed or interrupted trigger may already exist in Hatchet. Keep
        # the accepted run recoverable by an idempotent retry or maintenance.
        try:
            return await _submit_run(self._workflow, self.store, accepted)
        except Exception:
            logger.exception(
                "background.hatchet_submission_failed",
                extra={"response_id": accepted.response_id},
            )
            return accepted

    async def retrieve(self, response_id: str, owner_scope: str) -> StoredRun | None:
        """Read one authorized Response snapshot."""
        return await self.store.get(response_id, owner_scope)

    async def cancel(
        self,
        response_id: str,
        owner_scope: str,
        response: dict[str, JsonValue],
        *,
        stored: bool,
    ) -> StoredRun | None:
        """Choose the public cancellation first, then cancel its Hatchet run."""
        cancelled = await self.store.request_cancellation(
            response_id,
            owner_scope,
            response,
            now=datetime.now(UTC),
            result_retention=self.settings.result_retention_for(stored=stored),
            idempotency_retention=self.settings.idempotency_retention,
        )
        if (
            cancelled is not None
            and cancelled.status is ResponseStatus.CANCELLED
            and cancelled.cancellation_pending
        ):
            try:
                await _deliver_cancellation(self._runs, self.store, cancelled)
            except Exception:
                logger.exception(
                    "background.hatchet_cancellation_failed",
                    extra={"response_id": cancelled.response_id},
                )
        return cancelled


def create_hatchet_workflows(
    hatchet: Hatchet,
    worker: BackgroundJobExecutor,
    *,
    settings: HatchetAdapterSettings | None = None,
) -> HatchetWorkflows:
    """Register execution, final failure, and maintenance directly in Hatchet."""
    task_settings = settings or HatchetAdapterSettings()
    workflow = hatchet.workflow(
        name=task_settings.workflow_name,
        input_validator=HatchetResponseInput,
        idempotency=TTLBasedIdempotencyConfig(
            key_expression="input.response_id",
            ttl=task_settings.idempotency_ttl,
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
        _context: Context,
    ) -> dict[str, str]:
        await worker.execute(job_input.response_id)
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
        _context: Context,
    ) -> dict[str, str]:
        await worker.finalize(job_input.response_id)
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
        _context: Context,
    ) -> dict[str, int]:
        submitted = await _submit_pending_runs(worker, workflow)
        cancelled = await _deliver_pending_cancellations(
            worker,
            hatchet.runs,
        )
        return {
            **await worker.maintain(),
            "submitted": submitted,
            "cancelled": cancelled,
        }

    return HatchetWorkflows(response=workflow, maintenance=maintain)


async def _submit_run(
    workflow: HatchetResponseWorkflow,
    store: ResponseStore,
    run: StoredRun,
) -> StoredRun:
    task_input = HatchetResponseInput(response_id=run.response_id)
    try:
        reference = await workflow.aio_run(
            input=task_input,
            wait_for_result=False,
        )
        workflow_run_id = reference.workflow_run_id
    except IdempotencyCollisionError as exc:
        workflow_run_id = exc.existing_run_external_id

    with CancelScope(shield=True):
        recorded = await store.record_workflow_run(
            run.response_id,
            workflow_run_id,
            now=datetime.now(UTC),
        )
    if recorded is None:
        msg = "Hatchet workflow receipt could not be persisted."
        raise RuntimeError(msg)
    return recorded


async def _submit_pending_runs(
    worker: BackgroundJobExecutor,
    workflow: HatchetResponseWorkflow,
) -> int:
    pending = await worker.store.claim_pending_submissions(
        now=datetime.now(UTC),
        limit=worker.settings.maintenance_batch_size,
    )
    submitted = 0
    for run in pending:
        try:
            await _submit_run(workflow, worker.store, run)
            submitted += 1
        except Exception:
            logger.exception(
                "background.hatchet_submission_failed",
                extra={"response_id": run.response_id},
            )
    return submitted


async def _deliver_cancellation(
    runs: RunsClient,
    store: ResponseStore,
    run: StoredRun,
) -> bool:
    if not run.cancellation_pending or run.workflow_run_id is None:
        return False
    await runs.aio_cancel(run.workflow_run_id)
    return await store.finish_cancellation(
        run.response_id,
        now=datetime.now(UTC),
    )


async def _deliver_pending_cancellations(
    worker: BackgroundJobExecutor,
    runs: RunsClient,
) -> int:
    pending = await worker.store.claim_cancellations(
        now=datetime.now(UTC),
        limit=worker.settings.maintenance_batch_size,
    )
    delivered = 0
    for run in pending:
        try:
            delivered += int(await _deliver_cancellation(runs, worker.store, run))
        except Exception:
            logger.exception(
                "background.hatchet_cancellation_failed",
                extra={"response_id": run.response_id},
            )
    return delivered


async def check_hatchet_connection(hatchet: Hatchet) -> str | None:
    """Perform the adapter's read-only authentication/version connectivity check."""
    return await hatchet.aio_get_engine_version()


__all__ = [
    "BackgroundJobExecutor",
    "HatchetAdapterSettings",
    "HatchetBackgroundBackend",
    "HatchetMaintenanceTask",
    "HatchetResponseInput",
    "HatchetResponseWorkflow",
    "HatchetWorkflows",
    "check_hatchet_connection",
    "create_hatchet_workflows",
]
