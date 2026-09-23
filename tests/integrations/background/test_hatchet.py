import base64
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from hatchet_sdk import Hatchet
from hatchet_sdk.clients.rest.models.v1_task_status import V1TaskStatus
from hatchet_sdk.config import ClientConfig
from hatchet_sdk.exceptions import IdempotencyCollisionError
from hatchet_sdk.runnables.types import EmptyModel

from langgraph_openai_serve import BackgroundWorker, InMemoryResponseStore
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background.responses import queued_response, response_json
from langgraph_openai_serve.background.store import NewRun
from langgraph_openai_serve.integrations.background.hatchet import (
    HatchetAdapterSettings,
    HatchetBackgroundBackend,
    HatchetResponseInput,
    create_hatchet_workflows,
)


def _new_run() -> NewRun:
    now = datetime.now(UTC)
    envelope = {"model": "model", "input": "hello", "background": True}
    response = queued_response(
        ResponseCreateRequest.model_validate(envelope),
        response_id="resp_one",
        created_at=now.timestamp(),
    )
    return NewRun(
        response_id="resp_one",
        owner_scope="owner",
        model="model",
        checkpoint_thread_id="checkpoint",
        graph_version="v1",
        envelope=envelope,
        response=response_json(response),
        created_at=now,
    )


def _backend(
    *,
    workflow: Mock | None = None,
    runs: Mock | None = None,
) -> HatchetBackgroundBackend:
    return HatchetBackgroundBackend(
        workflow=workflow or Mock(aio_run=AsyncMock()),
        runs=runs or Mock(aio_bulk_cancel=AsyncMock()),
        store=InMemoryResponseStore(),
    )


def test_workflows_register_with_the_pinned_hatchet_sdk() -> None:
    claims = (
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "sub": "00000000-0000-0000-0000-000000000000",
                    "server_url": "http://localhost:8080",
                    "grpc_broadcast_address": "localhost:7070",
                }
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    client = Hatchet(
        config=ClientConfig(token=f"eyJhbGciOiJub25lIn0.{claims}.signature")
    )

    workflows = create_hatchet_workflows(client)

    assert workflows.response.name == "lgos-background-response"
    assert workflows.maintenance.name == "lgos-background-maintenance"
    assert {task.name for task in workflows.response.tasks} == {
        "lgos-background-response-execute",
        "lgos-background-response-finalize-on-failure",
    }


async def test_submit_triggers_the_workflow_with_response_metadata() -> None:
    workflow = Mock(aio_run=AsyncMock())
    backend = _backend(workflow=workflow)
    run = await backend.store.create(_new_run())

    await backend.submit(run)

    workflow.aio_run.assert_awaited_once_with(
        HatchetResponseInput(response_id="resp_one"),
        wait_for_result=False,
        additional_metadata={"lgos_response_id": "resp_one"},
    )


async def test_submit_treats_an_idempotency_collision_as_submitted() -> None:
    workflow = Mock(
        aio_run=AsyncMock(side_effect=IdempotencyCollisionError("hatchet-run"))
    )
    backend = _backend(workflow=workflow)
    run = await backend.store.create(_new_run())

    await backend.submit(run)

    workflow.aio_run.assert_awaited_once()


async def test_stop_cancels_active_hatchet_runs_by_response_metadata() -> None:
    runs = Mock(aio_bulk_cancel=AsyncMock())
    backend = _backend(runs=runs)
    run = await backend.store.create(_new_run())

    await backend.stop(run)

    (options,), _ = runs.aio_bulk_cancel.await_args
    assert options.filters.additional_metadata == {"lgos_response_id": "resp_one"}
    assert options.filters.since == run.created_at
    assert set(options.filters.statuses) == {
        V1TaskStatus.QUEUED,
        V1TaskStatus.RUNNING,
    }


class _FakeWorkflow:
    def __init__(self) -> None:
        self.task_options: dict[str, object] = {}
        self.failure_options: dict[str, object] = {}
        self.execute = None
        self.finalize = None

    def task(self, **options):
        self.task_options = options

        def decorator(function):
            self.execute = function
            return function

        return decorator

    def on_failure_task(self, **options):
        self.failure_options = options

        def decorator(function):
            self.finalize = function
            return function

        return decorator


async def test_registration_runs_the_lifespan_worker_with_native_options() -> None:
    execution_retries = 4
    finalization_retries = 2
    workflow = _FakeWorkflow()
    maintenance_options: dict[str, object] = {}
    maintenance_function = None
    hatchet = Mock(workflow=Mock(return_value=workflow))

    def register_maintenance(**options):
        maintenance_options.update(options)

        def decorator(function):
            nonlocal maintenance_function
            maintenance_function = function
            return function

        return decorator

    hatchet.task = register_maintenance
    # Unknown Response IDs finish before the worker needs a graph.
    worker = BackgroundWorker(graphs=Mock(), store=InMemoryResponseStore())
    settings = HatchetAdapterSettings(
        retries=execution_retries,
        finalization_retries=finalization_retries,
        maintenance_cron="*/10 * * * *",
    )

    registrations = create_hatchet_workflows(hatchet, settings=settings)
    job_input = HatchetResponseInput(response_id="resp_missing")
    context = Mock(lifespan=worker)
    assert workflow.execute is not None
    assert workflow.finalize is not None
    assert maintenance_function is not None

    assert await workflow.execute(job_input, context) == {"response_id": "resp_missing"}
    assert await workflow.finalize(job_input, context) == {
        "response_id": "resp_missing"
    }
    assert await maintenance_function(EmptyModel(), context) == {
        "resubmitted": 0,
        "cleaned": 0,
        "expired": 0,
    }
    with pytest.raises(TypeError, match="BackgroundWorker"):
        await workflow.execute(job_input, Mock(lifespan=None))
    assert registrations.response is workflow
    idempotency = hatchet.workflow.call_args.kwargs["idempotency"]
    assert idempotency.key_expression == "input.response_id"
    assert idempotency.ttl == settings.max_queue_time
    assert workflow.task_options["retries"] == execution_retries
    assert workflow.failure_options["retries"] == finalization_retries
    assert workflow.failure_options["schedule_timeout"] == settings.schedule_timeout
    assert maintenance_options["on_crons"] == ["*/10 * * * *"]
    assert maintenance_options["concurrency"] == 1


def test_max_queue_time_covers_every_schedule_attempt() -> None:
    settings = HatchetAdapterSettings(
        retries=2,
        schedule_timeout=timedelta(minutes=10),
        backoff_max_seconds=30,
    )

    assert settings.max_queue_time == timedelta(minutes=31, seconds=30)
