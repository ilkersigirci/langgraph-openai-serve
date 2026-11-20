"""Background Responses executed by a pluggable engine."""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from anyio import CancelScope, Event, create_task_group
from openai.types.responses import Response, ResponseError, ResponseStreamEvent
from pydantic import BaseModel, ConfigDict, JsonValue

from langgraph_openai_serve.api.responses.interrupts import interrupt_response_id
from langgraph_openai_serve.api.responses.output import ResponseContext
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import (
    EventPublisher,
    collect_response,
    prepare_response_run,
)
from langgraph_openai_serve.core.errors import InvalidRequestError
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.graph_registry import GraphRegistry

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = get_logger(__name__)

BackgroundStatus = Literal["queued", "in_progress", "completed", "failed", "cancelled"]


class BackgroundJob(BaseModel):
    """A validated request and the server-owned context that executes it."""

    request: ResponseCreateRequest
    owner_scope: str
    # Interrupt identity shared by every Response that continues one run.
    run_id: str
    created_at: int
    # Scoped Idempotency-Key digest, or a unique value when the client sent none.
    idempotency_key: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class BackgroundRun(BaseModel):
    """An engine's current view of one submitted job."""

    # The engine's run UUID; the public Response ID embeds it.
    id: str
    job: BackgroundJob
    status: BackgroundStatus
    # The executed Response, present once the engine run completed.
    response: dict[str, JsonValue] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def response_id(self) -> str:
        """Public ID that locates this run and continues its interrupts."""
        return interrupt_response_id(self.job.run_id, self.id)

    def queued_snapshot(self) -> Response:
        """Return the public Response as it was accepted."""
        return _context(self.job, self.id).response(status="queued", output=[])

    def snapshot(self) -> Response:
        """Return the public Response for the run's current state."""
        if self.response is not None:
            return Response.model_validate(self.response)
        response = self.queued_snapshot()
        # A finished run without a stored Response has no result to return.
        if self.status in {"completed", "failed"}:
            return _failed_response(response, "The background response failed.")
        return response.model_copy(update={"status": self.status})


class BackgroundBackend(Protocol):
    """An engine that runs background jobs and stores their status and result."""

    async def submit(self, job: BackgroundJob) -> BackgroundRun:
        """Start ``job``, or return the run already holding its idempotency key."""
        ...

    async def get(self, run_id: str) -> BackgroundRun | None:
        """Read one run, or return None when the engine does not have it."""
        ...

    async def cancel(self, run_id: str) -> None:
        """Stop one run; a finished run keeps its outcome."""
        ...

    def events(self, run_id: str) -> AsyncGenerator[str, None]:
        """
        Yield a run's published events from the first, then those that follow.

        Ending early is allowed: readers take the outcome from the run's status.
        """
        ...


async def execute_background_job(
    job: BackgroundJob,
    engine_run_id: str,
    registry: GraphRegistry,
    publish: EventPublisher | None = None,
) -> dict[str, JsonValue]:
    """
    Execute one job through the foreground Responses path.

    The request selects the graph's streaming mode. An engine can pass
    ``publish`` to store the events as well as the final Response.
    """
    context = _context(job, engine_run_id)
    try:
        run = await prepare_response_run(
            job.request,
            registry,
            checkpoint_scope=job.owner_scope,
            run_id=job.run_id,
        )
        response = await collect_response(job.request, run, context, publish=publish)
    except InvalidRequestError as exc:
        # The foreground path reports these as 4xx; a background Response
        # reports them as its own failure. An interrupt answer, for example,
        # is validated only once its run holds the lease, so a stale answer
        # fails here.
        queued = context.response(status="queued", output=[])
        response = _failed_response(queued, str(exc))
    return response.model_dump(mode="json", by_alias=True)


@dataclass
class _MemoryRun:
    current: BackgroundRun
    scope: CancelScope = field(default_factory=CancelScope)
    events: list[str] = field(default_factory=list)
    published: Event = field(default_factory=Event)

    async def publish(self, event: ResponseStreamEvent) -> None:
        self.events.append(event.model_dump_json(by_alias=True))
        self.published.set()
        self.published = Event()

    def transition(
        self, status: BackgroundStatus, response: dict[str, JsonValue] | None = None
    ) -> None:
        # The first terminal outcome wins, as in a durable engine.
        if self.current.status in {"queued", "in_progress"}:
            self.current = self.current.model_copy(
                update={"status": status, "response": response}
            )


class InMemoryBackgroundBackend:
    """
    Run background jobs as tasks of this application process.

    For development and tests: runs stay in memory until the process exits.
    """

    def __init__(self, registry: GraphRegistry) -> None:
        self._registry = registry
        self._runs: dict[str, _MemoryRun] = {}
        self._keys: dict[str, str] = {}
        self._tasks: TaskGroup | None = None

    @asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncGenerator[None, None]:
        """Own the execution tasks for one ASGI application lifespan."""
        async with create_task_group() as tasks:
            self._tasks = tasks
            try:
                yield
            finally:
                self._tasks = None
                tasks.cancel()

    async def submit(self, job: BackgroundJob) -> BackgroundRun:
        """Start one task, or return the run holding the job's key."""
        if self._tasks is None:
            msg = "The in-memory background backend lifespan is not running."
            raise RuntimeError(msg)
        if (run_id := self._keys.get(job.idempotency_key)) is not None:
            return self._runs[run_id].current
        run = BackgroundRun(id=str(uuid.uuid4()), job=job, status="queued")
        entry = _MemoryRun(run)
        self._runs[run.id] = entry
        self._keys[job.idempotency_key] = run.id
        self._tasks.start_soon(self._execute, entry)
        return run

    async def get(self, run_id: str) -> BackgroundRun | None:
        """Read one run."""
        entry = self._runs.get(run_id)
        return entry.current if entry is not None else None

    async def cancel(self, run_id: str) -> None:
        """Mark an active run cancelled and cancel its task."""
        if (entry := self._runs.get(run_id)) is not None:
            entry.transition("cancelled")
            entry.scope.cancel()

    async def events(self, run_id: str) -> AsyncGenerator[str, None]:
        """
        Yield the run's published events, waiting for new ones.

        Yields:
            Each encoded event, from the first.

        """
        entry = self._runs[run_id]
        position = 0
        while True:
            while position < len(entry.events):
                yield entry.events[position]
                position += 1
            await entry.published.wait()

    async def _execute(self, entry: _MemoryRun) -> None:
        run = entry.current
        with entry.scope:
            entry.transition("in_progress")
            try:
                response = await execute_background_job(
                    run.job,
                    run.id,
                    self._registry,
                    entry.publish if run.job.request.stream else None,
                )
            except Exception:
                logger.exception(
                    "background.in_memory_execution_failed",
                    extra={"response_id": run.response_id},
                )
                entry.transition("failed")
            else:
                entry.transition("completed", response)


def _context(job: BackgroundJob, engine_run_id: str) -> ResponseContext:
    """Return the identity that every snapshot of one background Response keeps."""
    return ResponseContext(
        request=job.request,
        id=interrupt_response_id(job.run_id, engine_run_id),
        created_at=job.created_at,
    )


def _failed_response(response: Response, message: str) -> Response:
    # ResponseError.code is OpenAI's closed vocabulary; server_error is the
    # only generic value.
    return response.model_copy(
        update={
            "status": "failed",
            "output": [],
            "error": ResponseError(code="server_error", message=message),
        }
    )
