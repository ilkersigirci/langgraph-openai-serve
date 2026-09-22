import base64
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from anyio import Event, create_task_group, fail_after
from hatchet_sdk import Hatchet
from hatchet_sdk.config import ClientConfig
from hatchet_sdk.exceptions import IdempotencyCollisionError
from hatchet_sdk.runnables.types import EmptyModel

from langgraph_openai_serve import BackgroundSettings, InMemoryResponseStore
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background.responses import active_response, response_json
from langgraph_openai_serve.background.store import NewRun, ResponseStatus
from langgraph_openai_serve.integrations import hatchet
from langgraph_openai_serve.integrations.hatchet import (
    HatchetAdapterSettings,
    HatchetBackgroundBackend,
    HatchetResponseInput,
    create_hatchet_workflows,
)


def _new_run(
    *,
    name: str = "one",
    idempotency_digest: str | None = None,
) -> NewRun:
    now = datetime.now(UTC)
    response_id = f"resp_{name}"
    envelope = {
        "model": "model",
        "input": "hello",
        "background": True,
        "stream": False,
    }
    response = active_response(
        ResponseCreateRequest.model_validate(envelope),
        response_id=response_id,
        created_at=now.timestamp(),
    )
    return NewRun(
        response_id=response_id,
        owner_scope="owner",
        model="model",
        checkpoint_thread_id="checkpoint",
        graph_version="v1",
        envelope=envelope,
        request_fingerprint="fingerprint",
        idempotency_digest=idempotency_digest,
        response=response_json(response),
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
    store = InMemoryResponseStore()
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=store,
    )

    created = await backend.create(_new_run())

    assert created.workflow_run_id == "native-run"
    workflow.aio_run.assert_awaited_once_with(
        input=HatchetResponseInput(response_id="resp_one"),
        wait_for_result=False,
    )


async def test_backend_uses_hatchet_idempotency_collision_receipt() -> None:
    workflow = Mock(
        aio_run=AsyncMock(side_effect=IdempotencyCollisionError("existing-native-run"))
    )
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=InMemoryResponseStore(),
    )

    created = await backend.create(_new_run())

    assert created.workflow_run_id == "existing-native-run"


async def test_failed_submission_remains_recoverable() -> None:
    store = InMemoryResponseStore()
    workflow = Mock(aio_run=AsyncMock(side_effect=OSError("unavailable")))
    backend = HatchetBackgroundBackend(
        workflow=workflow,
        runs=Mock(),
        store=store,
    )

    created = await backend.create(_new_run())

    pending = await store.get_internal("resp_one")

    assert pending is not None
    assert pending.status is ResponseStatus.QUEUED
    assert pending.workflow_run_id is None
    assert pending.result_expires_at is None
    assert created == pending


async def test_idempotent_retry_recovers_an_ambiguous_submission() -> None:
    expected_submissions = 2
    store = InMemoryResponseStore()
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
        settings=BackgroundSettings(admission_capacity=1),
    )
    run = _new_run(idempotency_digest="stable-digest")
    retry = _new_run(name="retry", idempotency_digest="stable-digest")

    accepted = await backend.create(run)
    retried = await backend.create(retry)

    assert accepted.response_id == run.response_id
    assert accepted.workflow_run_id is None
    assert retried.response_id == run.response_id
    assert retried.status is ResponseStatus.QUEUED
    assert retried.workflow_run_id == "native-run"
    assert workflow.aio_run.await_count == expected_submissions


async def test_failed_duplicate_submission_cannot_fail_a_submitted_run() -> None:
    started = Event()
    release = Event()
    submissions = []

    async def submit(**_kwargs):
        submissions.append(None)
        if len(submissions) == 1:
            started.set()
            await release.wait()
            message = "acknowledgement lost"
            raise OSError(message)
        return SimpleNamespace(workflow_run_id="native-run")

    store = InMemoryResponseStore()
    backend = HatchetBackgroundBackend(
        workflow=Mock(aio_run=submit), runs=Mock(), store=store
    )
    run = _new_run(idempotency_digest="stable-digest")

    delayed_results = []

    async def delayed_create():
        delayed_results.append(await backend.create(run))

    with fail_after(2):
        async with create_task_group() as tasks:
            tasks.start_soon(delayed_create)
            await started.wait()
            acknowledged = await backend.create(run)
            release.set()

    current = await backend.retrieve(run.response_id, run.owner_scope)
    assert delayed_results[0].workflow_run_id is None
    assert acknowledged.workflow_run_id == "native-run"
    assert current == acknowledged


@pytest.mark.parametrize("already_submitted", [False, True])
async def test_maintenance_recovers_an_interrupted_submission(already_submitted):
    store = InMemoryResponseStore()
    run = _new_run()
    # The API process disappears after acceptance or after an unrecorded trigger.
    await store.accept(run, capacity=1)
    workflow = _FakeWorkflow()
    workflow.aio_run = AsyncMock(
        side_effect=(
            IdempotencyCollisionError("native-run") if already_submitted else None
        ),
        return_value=SimpleNamespace(workflow_run_id="native-run"),
    )

    async def maintain():
        current = await store.get_internal(run.response_id)
        assert current is not None
        assert current.workflow_run_id == "native-run"
        return {"cleaned": 0, "expired": 0}

    worker = Mock(
        store=store,
        settings=BackgroundSettings(),
        maintain=AsyncMock(side_effect=maintain),
    )
    client = Mock(
        workflow=Mock(return_value=workflow),
        task=lambda **_options: lambda function: function,
    )
    registrations = create_hatchet_workflows(client, worker)

    result = await registrations.maintenance(EmptyModel(), Mock())
    recovered = await store.get(run.response_id, run.owner_scope)

    assert result["submitted"] == 1
    assert recovered is not None
    assert recovered.status is ResponseStatus.QUEUED
    assert recovered.workflow_run_id == "native-run"
    await registrations.maintenance(EmptyModel(), Mock())
    workflow.aio_run.assert_awaited_once()


async def test_pending_submission_failure_does_not_starve_later_runs():
    store = InMemoryResponseStore()
    for name in ("first", "second"):
        await store.accept(_new_run(name=name), capacity=2)

    async def submit(**kwargs):
        if kwargs["input"].response_id == "resp_first":
            message = "temporarily unavailable"
            raise OSError(message)
        return SimpleNamespace(workflow_run_id="native-second")

    worker = Mock(
        store=store,
        settings=BackgroundSettings(maintenance_batch_size=1),
    )
    workflow = Mock(aio_run=submit)

    assert await hatchet._submit_pending_runs(worker, workflow) == 0
    assert await hatchet._submit_pending_runs(worker, workflow) == 1
    first = await store.get_internal("resp_first")
    second = await store.get_internal("resp_second")
    assert first is not None
    assert first.status is ResponseStatus.QUEUED
    assert second is not None
    assert second.workflow_run_id == "native-second"


async def test_backend_cancellation_uses_native_hatchet_run() -> None:
    runs = Mock(aio_cancel=AsyncMock())
    backend = HatchetBackgroundBackend(
        workflow=Mock(
            aio_run=AsyncMock(
                return_value=SimpleNamespace(workflow_run_id="native-run")
            )
        ),
        runs=runs,
        store=InMemoryResponseStore(),
        settings=BackgroundSettings(),
    )
    await backend.create(_new_run())

    cancelled = await backend.cancel(
        "resp_one",
        "owner",
        {"id": "resp_one", "status": "cancelled"},
        stored=False,
    )

    assert cancelled is not None
    assert cancelled.status is ResponseStatus.CANCELLED
    assert cancelled.cancellation_pending is True
    runs.aio_cancel.assert_awaited_once_with("native-run")
    persisted = await backend.store.get_internal(cancelled.response_id)
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
        store=InMemoryResponseStore(),
    )
    await backend.create(_new_run())
    response = {"id": "resp_one", "status": "cancelled"}

    first = await backend.cancel("resp_one", "owner", response, stored=False)
    second = await backend.cancel("resp_one", "owner", response, stored=False)

    assert first is not None
    assert second is not None
    assert first.status is second.status is ResponseStatus.CANCELLED
    assert runs.aio_cancel.await_count == expected_attempts


async def test_backend_cancellation_uses_stored_result_retention() -> None:
    settings = BackgroundSettings(
        result_retention=timedelta(seconds=1),
        stored_result_retention=timedelta(days=30),
    )
    backend = HatchetBackgroundBackend(
        workflow=Mock(
            aio_run=AsyncMock(
                return_value=SimpleNamespace(workflow_run_id="native-run")
            )
        ),
        runs=Mock(aio_cancel=AsyncMock()),
        store=InMemoryResponseStore(),
        settings=settings,
    )
    await backend.create(_new_run())

    cancelled = await backend.cancel(
        "resp_one",
        "owner",
        {"id": "resp_one", "status": "cancelled"},
        stored=True,
    )

    assert cancelled is not None
    assert cancelled.result_expires_at is not None
    assert cancelled.terminal_at is not None
    assert cancelled.result_expires_at - cancelled.terminal_at == timedelta(days=30)


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
    store = InMemoryResponseStore()
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
    job_input = HatchetResponseInput(response_id="resp_one")
    assert workflow.execute is not None
    assert workflow.finalize is not None
    assert maintenance_function is not None
    await workflow.execute(job_input, Mock())
    await workflow.finalize(job_input, Mock())
    result = await maintenance_function(EmptyModel(), Mock())

    worker.execute.assert_awaited_once_with("resp_one")
    worker.finalize.assert_awaited_once_with("resp_one")
    worker.maintain.assert_awaited_once_with()
    assert result == {"cleaned": 1, "expired": 2, "submitted": 0, "cancelled": 0}
    assert workflow.task_options["retries"] == execution_retries
    assert workflow.failure_options["retries"] == finalization_retries
    assert workflow.failure_options["schedule_timeout"] == settings.schedule_timeout
    assert maintenance_options["on_crons"] == ["*/10 * * * *"]
    assert maintenance_options["concurrency"] == 1
    assert maintenance_options["schedule_timeout"] == settings.schedule_timeout
    idempotency = workflow_options["idempotency"]
    assert idempotency.key_expression == "input.response_id"
    assert registrations.response is workflow


async def test_maintenance_delivers_cancellation_after_transient_api_failure() -> None:
    expected_attempts = 2
    store = InMemoryResponseStore()
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
        stored=False,
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
    persisted = await store.get_internal(cancelled.response_id)

    assert result["cancelled"] == 1
    assert runs.aio_cancel.await_count == expected_attempts
    assert persisted is not None
    assert persisted.cancellation_pending is False


async def test_maintenance_resolves_receipt_before_cancelling() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    run = _new_run()
    await store.accept(run, capacity=1)
    cancelled = await store.request_cancellation(
        run.response_id,
        run.owner_scope,
        {"id": run.response_id, "status": "cancelled"},
        now=now,
        result_retention=timedelta(hours=1),
        idempotency_retention=timedelta(hours=1),
    )
    workflow = _FakeWorkflow()
    workflow.aio_run = AsyncMock(side_effect=IdempotencyCollisionError("native-run"))
    runs = Mock(aio_cancel=AsyncMock())
    worker = Mock(
        store=store,
        settings=BackgroundSettings(),
        maintain=AsyncMock(return_value={"cleaned": 0, "expired": 0}),
    )
    client = Mock(
        runs=runs,
        workflow=Mock(return_value=workflow),
        task=lambda **_options: lambda function: function,
    )

    registrations = create_hatchet_workflows(client, worker)
    result = await registrations.maintenance(EmptyModel(), Mock())
    persisted = await store.get_internal(run.response_id)

    assert cancelled is not None
    assert cancelled.cancellation_pending is True
    assert result["submitted"] == result["cancelled"] == 1
    runs.aio_cancel.assert_awaited_once_with("native-run")
    assert persisted is not None
    assert persisted.status is ResponseStatus.CANCELLED
    assert persisted.cancellation_pending is False


async def test_maintenance_continues_after_one_cancellation_fails() -> None:
    store = InMemoryResponseStore()
    first = await store.accept(_new_run(name="first"), capacity=2)
    second = await store.accept(_new_run(name="second"), capacity=2)
    await store.record_workflow_run(
        first.response_id,
        "native-first",
        now=datetime.now(UTC),
    )
    await store.record_workflow_run(
        second.response_id,
        "native-second",
        now=datetime.now(UTC),
    )
    for run in (first, second):
        await store.request_cancellation(
            run.response_id,
            "owner",
            {"id": run.response_id, "status": "cancelled"},
            now=datetime.now(UTC),
            result_retention=timedelta(hours=1),
            idempotency_retention=timedelta(hours=1),
        )

    async def cancel(workflow_run_id: str) -> None:
        if workflow_run_id == "native-first":
            message = "transient"
            raise OSError(message)

    runs = Mock(aio_cancel=AsyncMock(side_effect=cancel))
    worker = Mock(
        store=store,
        settings=BackgroundSettings(maintenance_batch_size=2),
    )

    delivered = await hatchet._deliver_pending_cancellations(worker, runs)
    first_persisted = await store.get_internal(first.response_id)
    second_persisted = await store.get_internal(second.response_id)

    assert delivered == 1
    assert runs.aio_cancel.await_args_list == [
        call("native-first"),
        call("native-second"),
    ]
    assert first_persisted is not None
    assert first_persisted.cancellation_pending is True
    assert second_persisted is not None
    assert second_persisted.cancellation_pending is False
