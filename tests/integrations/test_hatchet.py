import base64
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from hatchet_sdk import Hatchet
from hatchet_sdk.config import ClientConfig
from hatchet_sdk.exceptions import IdempotencyCollisionError
from hatchet_sdk.runnables.types import EmptyModel
from tests.background.fakes import MemoryResponseStore

from langgraph_openai_serve import BackgroundSettings, RunJob
from langgraph_openai_serve.background.store import NewRun, ResponseStatus
from langgraph_openai_serve.integrations.hatchet import (
    HatchetAdapterSettings,
    HatchetBackgroundBackend,
    HatchetRunInput,
    create_hatchet_workflows,
)


def _new_run(
    *,
    name: str = "one",
    idempotency_key: str | None = None,
) -> NewRun:
    now = datetime.now(UTC)
    response_id = f"resp_{name}"
    return NewRun(
        run_id=response_id,
        response_id=response_id,
        owner_scope="owner",
        model="model",
        checkpoint_thread_id="checkpoint",
        graph_version="v1",
        envelope={
            "model": "model",
            "input": "hello",
            "background": True,
            "stream": False,
        },
        request_fingerprint="fingerprint",
        idempotency_key=idempotency_key,
        response={"id": response_id, "status": "queued"},
        created_at=now,
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
    worker = Mock(
        execute=AsyncMock(),
        finalize=AsyncMock(),
        maintain=AsyncMock(),
    )

    workflows = create_hatchet_workflows(client, worker)

    assert workflows.response.name == "lgos-background-response"
    assert workflows.maintenance.name == "lgos-background-maintenance"
    assert {task.name for task in workflows.response.tasks} == {
        "lgos-background-response-execute",
        "lgos-background-response-finalize-on-failure",
    }


async def test_backend_submits_without_waiting_and_persists_hatchet_id() -> None:
    workflow = Mock(
        aio_run=AsyncMock(return_value=SimpleNamespace(workflow_run_id="native-run"))
    )
    store = MemoryResponseStore()
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=store,
    )

    created = await backend.create(_new_run())

    assert created.workflow_run_id == "native-run"
    workflow.aio_run.assert_awaited_once_with(
        input=HatchetRunInput(run_id="resp_one"),
        wait_for_result=False,
    )


async def test_backend_uses_hatchet_idempotency_collision_receipt() -> None:
    workflow = Mock(
        aio_run=AsyncMock(side_effect=IdempotencyCollisionError("existing-native-run"))
    )
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=MemoryResponseStore(),
    )

    created = await backend.create(_new_run())

    assert created.workflow_run_id == "existing-native-run"


async def test_failed_non_idempotent_submission_discards_unacknowledged_row() -> None:
    store = MemoryResponseStore()
    workflow = Mock(aio_run=AsyncMock(side_effect=OSError("unavailable")))
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=store,
    )

    with pytest.raises(OSError, match="unavailable"):
        await backend.create(_new_run())

    assert await store.get_internal("resp_one") is None


async def test_failed_idempotent_submission_keeps_stable_run_for_retry() -> None:
    store = MemoryResponseStore()
    workflow = Mock(
        aio_run=AsyncMock(
            side_effect=[
                OSError("acknowledgement lost"),
                IdempotencyCollisionError("native-run"),
            ]
        )
    )
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=store,
    )
    run = _new_run(idempotency_key="stable-key")
    retry = _new_run(name="retry", idempotency_key="stable-key")

    with pytest.raises(OSError, match="acknowledgement lost"):
        await backend.create(run)
    pending = await store.get_internal(run.run_id)
    retried = await backend.create(retry)

    assert pending is not None
    assert pending.workflow_run_id is None
    assert retried.run_id == run.run_id
    assert retried.workflow_run_id == "native-run"


async def test_backend_cancellation_uses_native_hatchet_run() -> None:
    runs = Mock(aio_cancel=AsyncMock())
    backend = HatchetBackgroundBackend(
        workflow=Mock(
            aio_run=AsyncMock(
                return_value=SimpleNamespace(workflow_run_id="native-run")
            )
        ),
        runs=runs,
        store=MemoryResponseStore(),
        settings=BackgroundSettings(),
    )
    await backend.create(_new_run())

    cancelled = await backend.cancel(
        "resp_one",
        "owner",
        {"id": "resp_one", "status": "cancelled"},
    )

    assert cancelled is not None
    assert cancelled.status is ResponseStatus.CANCELLED
    assert cancelled.cancellation_pending is True
    runs.aio_cancel.assert_awaited_once_with("native-run")
    persisted = await backend.store.get_internal(cancelled.run_id)
    assert persisted is not None
    assert persisted.cancellation_pending is False


async def test_repeated_cancellation_retries_a_transient_hatchet_failure() -> None:
    expected_attempts = 2
    message = "transient"
    runs = Mock(aio_cancel=AsyncMock(side_effect=[OSError(message), None]))
    backend = HatchetBackgroundBackend(
        workflow=Mock(
            aio_run=AsyncMock(
                return_value=SimpleNamespace(workflow_run_id="native-run")
            )
        ),
        runs=runs,
        store=MemoryResponseStore(),
    )
    await backend.create(_new_run())
    response = {"id": "resp_one", "status": "cancelled"}

    first = await backend.cancel("resp_one", "owner", response)
    second = await backend.cancel("resp_one", "owner", response)

    assert first is not None
    assert second is not None
    assert first.status is second.status is ResponseStatus.CANCELLED
    assert runs.aio_cancel.await_count == expected_attempts


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


async def test_registration_delegates_retries_failure_and_cron_to_hatchet() -> None:
    execution_retries = 4
    finalization_retries = 2
    workflow = _FakeWorkflow()
    workflow_options: dict[str, object] = {}
    maintenance_options: dict[str, object] = {}
    maintenance_function = None
    store = MemoryResponseStore()
    runs = Mock(aio_cancel=AsyncMock())
    hatchet = Mock(runs=runs)

    def register_workflow(**options):
        workflow_options.update(options)
        return workflow

    def register_maintenance(**options):
        maintenance_options.update(options)

        def decorator(function):
            nonlocal maintenance_function
            maintenance_function = function
            return function

        return decorator

    hatchet.workflow = register_workflow
    hatchet.task = register_maintenance
    worker = Mock(
        execute=AsyncMock(),
        finalize=AsyncMock(),
        maintain=AsyncMock(return_value={"cleaned": 1, "expired": 2}),
        store=store,
        settings=BackgroundSettings(),
    )
    settings = HatchetAdapterSettings(
        retries=execution_retries,
        finalization_retries=finalization_retries,
        maintenance_cron="*/10 * * * *",
    )

    registrations = create_hatchet_workflows(hatchet, worker, settings=settings)
    job_input = HatchetRunInput(run_id="run", schema_version=7)
    assert workflow.execute is not None
    assert workflow.finalize is not None
    assert maintenance_function is not None
    await workflow.execute(job_input, Mock())
    await workflow.finalize(job_input, Mock())
    result = await maintenance_function(EmptyModel(), Mock())

    worker.execute.assert_awaited_once_with(RunJob("run", schema_version=7))
    worker.finalize.assert_awaited_once_with(RunJob("run", schema_version=7))
    worker.maintain.assert_awaited_once_with()
    assert result == {"cleaned": 1, "expired": 2, "cancelled": 0}
    assert workflow.task_options["retries"] == execution_retries
    assert workflow.failure_options["retries"] == finalization_retries
    assert workflow.failure_options["schedule_timeout"] == settings.schedule_timeout
    assert maintenance_options["on_crons"] == ["*/10 * * * *"]
    assert maintenance_options["concurrency"] == 1
    assert maintenance_options["schedule_timeout"] == settings.schedule_timeout
    idempotency = workflow_options["idempotency"]
    assert idempotency.key_expression == "input.run_id"
    assert registrations.response is workflow


async def test_maintenance_delivers_cancellation_after_transient_api_failure() -> None:
    expected_attempts = 2
    store = MemoryResponseStore()
    runs = Mock(aio_cancel=AsyncMock(side_effect=[OSError("transient"), None]))
    workflow = Mock(
        aio_run=AsyncMock(return_value=SimpleNamespace(workflow_run_id="native-run"))
    )
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=runs,
        store=store,
    )
    await backend.create(_new_run())
    cancelled = await backend.cancel(
        "resp_one",
        "owner",
        {"id": "resp_one", "status": "cancelled"},
    )
    assert cancelled is not None

    native_workflow = _FakeWorkflow()
    maintenance_function = None
    hatchet = Mock(runs=runs)
    hatchet.workflow = Mock(return_value=native_workflow)

    def register_maintenance(**_options):
        def decorator(function):
            nonlocal maintenance_function
            maintenance_function = function
            return function

        return decorator

    hatchet.task = register_maintenance
    worker = Mock(
        execute=AsyncMock(),
        finalize=AsyncMock(),
        maintain=AsyncMock(return_value={"cleaned": 0, "expired": 0}),
        store=store,
        settings=BackgroundSettings(),
    )
    create_hatchet_workflows(hatchet, worker)
    assert maintenance_function is not None

    result = await maintenance_function(EmptyModel(), Mock())
    persisted = await store.get_internal(cancelled.run_id)

    assert result["cancelled"] == 1
    assert runs.aio_cancel.await_count == expected_attempts
    assert persisted is not None
    assert persisted.cancellation_pending is False
