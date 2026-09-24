"""Hatchet engine for background Responses."""

from datetime import timedelta

import grpc
from hatchet_sdk import (
    Context,
    Hatchet,
    IdempotencyCollisionError,
    RunStatus,
    TTLBasedIdempotencyConfig,
)
from hatchet_sdk.features.runs import RunsClient
from hatchet_sdk.runnables.workflow import Standalone
from pydantic import JsonValue, ValidationError

from langgraph_openai_serve.background import (
    BackgroundJob,
    BackgroundRun,
    BackgroundStatus,
    execute_background_job,
)
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

# A retried create within this window returns the run holding its key; Stripe
# keeps idempotency keys for the same 24 hours.
_IDEMPOTENCY_TTL = timedelta(days=1)

_STATUSES: dict[RunStatus, BackgroundStatus] = {
    RunStatus.QUEUED: "queued",
    RunStatus.RUNNING: "in_progress",
    RunStatus.COMPLETED: "completed",
    RunStatus.FAILED: "failed",
    RunStatus.CANCELLED: "cancelled",
}

HatchetBackgroundTask = Standalone[BackgroundJob, dict[str, JsonValue]]


def create_hatchet_task(
    hatchet: Hatchet,
    *,
    name: str = "lgos-background-response",
    schedule_timeout: timedelta = timedelta(minutes=30),
    execution_timeout: timedelta = timedelta(hours=1),
) -> HatchetBackgroundTask:
    """
    Register the background Response task in the API and worker processes.

    The worker lifespan must yield the ``GraphRegistry`` the task executes.
    The task is not retried: a failed Response is final, and the client may
    create a new one.
    """

    @hatchet.task(
        name=name,
        input_validator=BackgroundJob,
        schedule_timeout=schedule_timeout,
        execution_timeout=execution_timeout,
        idempotency=TTLBasedIdempotencyConfig(
            key_expression="input.idempotency_key",
            ttl=_IDEMPOTENCY_TTL,
        ),
    )
    async def run_background_response(
        job: BackgroundJob,
        context: Context,
    ) -> dict[str, JsonValue]:
        graphs = context.lifespan
        if not isinstance(graphs, GraphRegistry):
            msg = "The Hatchet worker lifespan must yield a GraphRegistry."
            raise TypeError(msg)
        return await execute_background_job(job, context.workflow_run_id, graphs)

    return run_background_response


class HatchetBackgroundBackend:
    """Submit, read, and cancel background Responses as Hatchet task runs."""

    def __init__(self, task: HatchetBackgroundTask, runs: RunsClient) -> None:
        self._task = task
        self._runs = runs

    async def submit(self, job: BackgroundJob) -> BackgroundRun:
        """Trigger the task, or return the run holding the job's key."""
        try:
            ref = await self._task.aio_run(job, wait_for_result=False)
        except IdempotencyCollisionError as exc:
            existing = await self.get(exc.existing_run_external_id)
            if existing is None:
                msg = "Hatchet reported an idempotency collision with no run."
                raise RuntimeError(msg) from exc
            return existing
        return BackgroundRun(id=ref.workflow_run_id, job=job, status="queued")

    async def get(self, run_id: str) -> BackgroundRun | None:
        """Read the run's input, status, and output from Hatchet."""
        try:
            details = await self._runs.aio_get_details(run_id)
        except grpc.RpcError as exc:
            if (
                isinstance(exc, (grpc.Call, grpc.aio.AioRpcError))
                and exc.code() == grpc.StatusCode.NOT_FOUND
            ):
                return None
            raise
        try:
            job = BackgroundJob.model_validate(details.input)
        except ValidationError:
            # A Response ID can name another run in the same Hatchet tenant.
            return None
        task_run = next(iter(details.task_runs.values()), None)
        completed = details.status is RunStatus.COMPLETED
        return BackgroundRun(
            id=run_id,
            job=job,
            status=_STATUSES[details.status],
            response=task_run.output if completed and task_run else None,
        )

    async def cancel(self, run_id: str) -> None:
        """Cancel the Hatchet run."""
        await self._runs.aio_cancel(run_id)


__all__ = [
    "HatchetBackgroundBackend",
    "HatchetBackgroundTask",
    "create_hatchet_task",
]
