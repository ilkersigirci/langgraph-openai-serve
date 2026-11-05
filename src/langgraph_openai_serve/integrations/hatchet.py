"""Hatchet engine for background Responses."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
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
from hatchet_sdk.features.streams import StreamsClient
from hatchet_sdk.runnables.workflow import Standalone
from openai.types.responses import ResponseStreamEvent, ResponseTextDeltaEvent
from pydantic import JsonValue, ValidationError

from langgraph_openai_serve.background import (
    BackgroundJob,
    BackgroundRun,
    BackgroundStatus,
    execute_background_job,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = get_logger(__name__)

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
"""The Hatchet task that ``create_hatchet_task`` registers."""


# How long a text delta may wait for later ones to share its publish request.
_DELTA_BATCH_TIMEOUT = timedelta(milliseconds=100)


def _topic(run_id: str) -> str:
    return f"lgos-response-{run_id}"


async def _first_to_publish(streams: StreamsClient, run_id: str) -> bool:
    """
    Return whether this attempt may publish its run's events.

    Hatchet reruns a task whose worker died, and that attempt starts over with its
    own items and sequence numbers. OpenAI's stream cannot tell a client to discard
    what it has, so a topic keeps one attempt, and a rerun's readers wait for its
    final Response. Only an engine that reports no such topic proves there was none.
    """
    try:
        await streams.aio_topic_metadata(_topic(run_id))
    except grpc.RpcError as exc:
        if _is_not_found(exc):
            return True
        logger.warning(
            "background.events_not_published",
            extra={"run_id": run_id},
            exc_info=True,
        )
    return False


class _TopicPublisher:
    """
    Publish one run's events to its durable stream.

    Streaming is best effort: the run must not fail because a tenant lacks durable
    streams or the engine refuses an event. The first failure ends publishing, and
    readers then get the final Response from the run's status.

    The engine limits publish requests per API token, and publishes share a request
    only while they wait together. So text deltas are not awaited and may wait for
    later ones. Any other event goes out at once with the deltas before it and is
    awaited, so the last event is stored before the run finishes. Every publish runs
    as a task, because the SDK orders a topic's messages by when each publish call
    begins, and tasks begin in the order they were created.
    """

    def __init__(self, streams: StreamsClient, run_id: str) -> None:
        self._streams = streams
        self._run_id = run_id
        # asyncio keeps only weak references to tasks.
        self._publishing: set[asyncio.Task[None]] = set()
        self._stopped = False

    async def __call__(self, event: ResponseStreamEvent) -> None:
        if self._stopped:
            return
        is_delta = isinstance(event, ResponseTextDeltaEvent)
        task = asyncio.create_task(
            self._streams.publish(
                _topic(self._run_id),
                event.model_dump_json(by_alias=True),
                batch_timeout=_DELTA_BATCH_TIMEOUT if is_delta else None,
            )
        )
        self._publishing.add(task)
        task.add_done_callback(self._published)
        if not is_delta:
            await asyncio.wait({task})

    def _published(self, task: asyncio.Task[None]) -> None:
        self._publishing.discard(task)
        error = None if task.cancelled() else task.exception()
        if error is not None and not self._stopped:
            self._stopped = True
            logger.warning(
                "background.events_not_published",
                extra={"run_id": self._run_id},
                exc_info=error,
            )


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
        registry = context.lifespan
        if not isinstance(registry, GraphRegistry):
            msg = "The Hatchet worker lifespan must yield a GraphRegistry."
            raise TypeError(msg)
        run_id = context.workflow_run_id
        streams = hatchet.streams
        publish = (
            _TopicPublisher(streams, run_id)
            if job.request.stream and await _first_to_publish(streams, run_id)
            else None
        )
        return await execute_background_job(job, run_id, registry, publish)

    return run_background_response


class HatchetBackgroundBackend:
    """Submit, read, cancel, and follow background Responses as Hatchet task runs."""

    def __init__(
        self,
        task: HatchetBackgroundTask,
        runs: RunsClient,
        streams: StreamsClient,
    ) -> None:
        self._task = task
        self._runs = runs
        self._streams = streams

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
            if _is_not_found(exc):
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

    async def events(self, run_id: str) -> AsyncGenerator[str, None]:
        """
        Yield the run's durable stream; a failure only ends it.

        The SDK reconnects by itself after a lost connection. The engine also
        hangs up a subscription that stayed idle for 30 minutes, which a queued or
        quiet run can outlast, so that resumes from the last event here.

        Yields:
            Each encoded event, from the first.

        """
        cursor = None
        try:
            while True:
                events = self._streams.subscribe(_topic(run_id), cursor=cursor)
                async with aclosing(events):
                    async for event in events:
                        cursor = event.cursor
                        yield event.payload.decode()
        except Exception:
            logger.warning(
                "background.events_not_read", extra={"run_id": run_id}, exc_info=True
            )


def _is_not_found(exc: grpc.RpcError) -> bool:
    return (
        isinstance(exc, (grpc.Call, grpc.aio.AioRpcError))
        and exc.code() == grpc.StatusCode.NOT_FOUND
    )


__all__ = [
    "HatchetBackgroundBackend",
    "HatchetBackgroundTask",
    "create_hatchet_task",
]
