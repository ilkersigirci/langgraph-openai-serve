"""Polling-only background Responses executed by a pluggable engine."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, Protocol

from anyio import CancelScope, create_task_group
from openai.types.responses import Response, ResponseError
from pydantic import BaseModel, ConfigDict, JsonValue

from langgraph_openai_serve.api.responses.interrupts import interrupt_response_id
from langgraph_openai_serve.api.responses.messages import InvalidResponsesInputError
from langgraph_openai_serve.api.responses.output import ResponseContext
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import (
    collect_response,
    prepare_response_run,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.client_settings import ClientSettingsValidationError
from langgraph_openai_serve.graph.graph_registry import GraphNotFoundError
from langgraph_openai_serve.graph.interrupt.coordination import RunBusyError
from langgraph_openai_serve.graph.interrupt.errors import InvalidResumeRequestError
from langgraph_openai_serve.graph.interrupt.state import (
    InterruptStateConflictError,
    InvalidRunIDError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anyio.abc import TaskGroup

    from langgraph_openai_serve.graph.graph_registry import GraphRegistry

logger = get_logger(__name__)

BackgroundStatus = Literal["queued", "in_progress", "completed", "failed", "cancelled"]

# Failures the foreground path reports as 4xx. A background Response reports
# them as its own failure. An interrupt answer, for example, is validated only
# once its run holds the checkpoint lease, so a stale answer fails there.
_REQUEST_ERRORS = (
    ClientSettingsValidationError,
    GraphNotFoundError,
    InterruptStateConflictError,
    InvalidResponsesInputError,
    InvalidResumeRequestError,
    InvalidRunIDError,
    RunBusyError,
    UnsupportedResponsesRequestError,
)


class BackgroundJob(BaseModel):
    """A validated request and the server-owned context that executes it."""

    request: dict[str, JsonValue]
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

    def snapshot(self) -> Response:
        """Return the public Response for the run's current state."""
        if self.response is not None:
            return Response.model_validate(self.response)
        response = _queued_response(self.job, self.response_id)
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


async def execute_background_job(
    job: BackgroundJob,
    engine_run_id: str,
    graphs: GraphRegistry,
) -> dict[str, JsonValue]:
    """Execute one job through the foreground Responses path."""
    request = ResponseCreateRequest.model_validate(job.request)
    response_id = interrupt_response_id(job.run_id, engine_run_id)
    try:
        run = await prepare_response_run(
            request,
            graphs,
            checkpoint_scope=job.owner_scope,
            run_id=job.run_id,
        )
        response = await collect_response(
            request,
            run,
            response_id=response_id,
            created_at=job.created_at,
        )
    except _REQUEST_ERRORS as exc:
        response = _failed_response(_queued_response(job, response_id), str(exc))
    return response.model_dump(mode="json", by_alias=True)


class InMemoryBackgroundBackend:
    """
    Run background jobs as tasks of this application process.

    For development and tests: runs stay in memory until the process exits.
    """

    def __init__(self, graphs: GraphRegistry) -> None:
        self._graphs = graphs
        self._runs: dict[str, BackgroundRun] = {}
        self._keys: dict[str, str] = {}
        self._scopes: dict[str, CancelScope] = {}
        self._tasks: TaskGroup | None = None

    @asynccontextmanager
    async def lifespan(self, _app: object) -> AsyncIterator[None]:
        """Own the execution tasks for one ASGI application lifespan."""
        async with create_task_group() as tasks:
            self._tasks = tasks
            try:
                yield
            finally:
                self._tasks = None
                tasks.cancel_scope.cancel()

    async def submit(self, job: BackgroundJob) -> BackgroundRun:
        """Start one task, or return the run holding the job's key."""
        if self._tasks is None:
            msg = "The in-memory background backend lifespan is not running."
            raise RuntimeError(msg)
        if (run_id := self._keys.get(job.idempotency_key)) is not None:
            return self._runs[run_id]
        run = BackgroundRun(id=str(uuid.uuid4()), job=job, status="queued")
        self._runs[run.id] = run
        self._keys[job.idempotency_key] = run.id
        self._scopes[run.id] = CancelScope()
        self._tasks.start_soon(self._execute, run)
        return run

    async def get(self, run_id: str) -> BackgroundRun | None:
        """Read one run."""
        return self._runs.get(run_id)

    async def cancel(self, run_id: str) -> None:
        """Mark an active run cancelled and cancel its task."""
        if run_id in self._runs:
            self._transition(run_id, "cancelled")
        if (scope := self._scopes.get(run_id)) is not None:
            scope.cancel()

    async def _execute(self, run: BackgroundRun) -> None:
        try:
            with self._scopes[run.id]:
                self._transition(run.id, "in_progress")
                try:
                    response = await execute_background_job(
                        run.job, run.id, self._graphs
                    )
                except Exception:
                    logger.exception(
                        "background.in_memory_execution_failed",
                        extra={"response_id": run.response_id},
                    )
                    self._transition(run.id, "failed")
                else:
                    self._transition(run.id, "completed", response)
        finally:
            del self._scopes[run.id]

    def _transition(
        self,
        run_id: str,
        status: BackgroundStatus,
        response: dict[str, JsonValue] | None = None,
    ) -> None:
        run = self._runs[run_id]
        # The first terminal outcome wins, as in a durable engine.
        if run.status in {"queued", "in_progress"}:
            self._runs[run_id] = run.model_copy(
                update={"status": status, "response": response}
            )


def _queued_response(job: BackgroundJob, response_id: str) -> Response:
    return ResponseContext.for_run(
        ResponseCreateRequest.model_validate(job.request),
        response_id=response_id,
        created_at=job.created_at,
    ).response(status="queued", output=[])


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


__all__ = [
    "BackgroundBackend",
    "BackgroundJob",
    "BackgroundRun",
    "BackgroundStatus",
    "InMemoryBackgroundBackend",
    "execute_background_job",
]
