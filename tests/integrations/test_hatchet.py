import base64
import json
import uuid
from types import SimpleNamespace

import grpc
import pytest
from hatchet_sdk import Hatchet
from hatchet_sdk.clients.admin import RunStatus, TaskRunDetail, WorkflowRunDetail
from hatchet_sdk.clients.rest.models.v1_task_status import V1TaskStatus
from hatchet_sdk.config import ClientConfig
from hatchet_sdk.exceptions import IdempotencyCollisionError
from tests.graph.support.message import make_message_graph

from langgraph_openai_serve import (
    BackgroundJob,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
)
from langgraph_openai_serve.integrations.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_task,
)

RUN_ID = str(uuid.uuid4())
JOB = BackgroundJob(
    request={"model": "background", "input": "Hello", "background": True},
    owner_scope="tenant-a",
    run_id=str(uuid.uuid4()),
    created_at=1_700_000_000,
    idempotency_key="key",
)


class _Task:
    """Captures the options and function the adapter registers."""

    def __init__(self, error: Exception | None = None) -> None:
        self.options: dict[str, object] = {}
        self.error = error

    def task(self, **options):
        self.options = options

        def register(function):
            self.run = function
            return self

        return register

    async def aio_run(self, job: BackgroundJob, *, wait_for_result: bool):
        assert wait_for_result is False
        if self.error is not None:
            raise self.error
        return SimpleNamespace(workflow_run_id=RUN_ID)


class _Runs:
    def __init__(self, *details: WorkflowRunDetail) -> None:
        self.details = {detail.external_id: detail for detail in details}
        self.cancelled: list[str] = []

    async def aio_get_details(self, run_id: str) -> WorkflowRunDetail:
        if run_id not in self.details:
            metadata = grpc.aio.Metadata()
            raise grpc.aio.AioRpcError(grpc.StatusCode.NOT_FOUND, metadata, metadata)
        return self.details[run_id]

    async def aio_cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)


def _completed(response: dict[str, object]) -> WorkflowRunDetail:
    return WorkflowRunDetail(
        external_id=RUN_ID,
        status=RunStatus.COMPLETED,
        input=JOB.model_dump(mode="json"),
        task_runs={
            "lgos-background-response": TaskRunDetail(
                external_id=str(uuid.uuid4()),
                readable_id="lgos-background-response",
                output=response,
                status=V1TaskStatus.COMPLETED,
            )
        },
        done=True,
    )


def test_task_registers_with_the_pinned_hatchet_sdk() -> None:
    claims = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "00000000-0000-0000-0000-000000000000",
                "server_url": "http://localhost:8080",
                "grpc_broadcast_address": "localhost:7070",
            }
        ).encode()
    )
    token = f"eyJhbGciOiJub25lIn0.{claims.decode().rstrip('=')}.signature"

    task = create_hatchet_task(Hatchet(config=ClientConfig(token=token)))

    assert task.name == "lgos-background-response"


async def test_task_executes_the_job_with_the_worker_lifespan_graphs() -> None:
    hatchet = _Task()
    create_hatchet_task(hatchet)  # ty: ignore[invalid-argument-type] - Captures the native registration.
    registry = GraphRegistry(
        registry={
            "background": GraphConfig(
                graph=make_message_graph("report"),
                description="Background test graph",
                features={GraphFeature.BACKGROUND},
            )
        }
    )
    context = SimpleNamespace(lifespan=registry, workflow_run_id=RUN_ID)

    response = await hatchet.run(JOB, context)

    # The key must name the job field that carries the scoped Idempotency-Key.
    assert hatchet.options["idempotency"].key_expression == "input.idempotency_key"
    assert response["status"] == "completed"
    assert response["id"].endswith(uuid.UUID(RUN_ID).hex)
    assert response["output"][0]["content"][0]["text"] == "report"
    with pytest.raises(TypeError, match="GraphRegistry"):
        await hatchet.run(JOB, SimpleNamespace(lifespan=None, workflow_run_id=RUN_ID))


async def test_backend_submits_reads_and_cancels_hatchet_runs() -> None:
    runs = _Runs(_completed({"id": "resp_done", "status": "completed"}))
    backend = HatchetBackgroundBackend(_Task(), runs)  # ty: ignore[invalid-argument-type] - Fakes of the native task and runs client.

    submitted = await backend.submit(JOB)
    completed = await backend.get(RUN_ID)
    missing = await backend.get(str(uuid.uuid4()))
    await backend.cancel(RUN_ID)

    assert (submitted.id, submitted.status) == (RUN_ID, "queued")
    assert completed is not None
    assert completed.job == JOB
    assert completed.response == {"id": "resp_done", "status": "completed"}
    assert missing is None
    assert runs.cancelled == [RUN_ID]


async def test_idempotency_collision_returns_the_existing_run() -> None:
    task = _Task(error=IdempotencyCollisionError(RUN_ID))
    backend = HatchetBackgroundBackend(
        task,  # ty: ignore[invalid-argument-type] - Fake of the native task.
        _Runs(_completed({"id": "resp_done", "status": "completed"})),  # ty: ignore[invalid-argument-type] - Fake of the runs client.
    )

    existing = await backend.submit(JOB)

    assert existing.status == "completed"
    assert existing.response == {"id": "resp_done", "status": "completed"}
