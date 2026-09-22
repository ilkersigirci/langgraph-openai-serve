"""Restart and race coverage for the PostgreSQL background Response store."""

import asyncio
import hashlib
import os
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langgraph_openai_serve import GraphRegistry
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import accept_background_response
from langgraph_openai_serve.background.store import (
    NewRun,
    ResponseStatus,
)
from langgraph_openai_serve.integrations.hatchet import HatchetBackgroundBackend

from lgos_demo_api.checkpointer import postgres_runtime, setup_postgres_schema
from lgos_demo_api.graphs.background_report import (
    create_background_report_config,
    create_background_report_graph,
)

POSTGRES_URI = os.environ.get(
    "DEMO_API_TEST_POSTGRES_URI",
    "postgresql://lgos:lgos@localhost:3001/lgos",
)

pytestmark = pytest.mark.integration


def _new_run(
    *,
    response_id: str,
    idempotency_digest: str,
    created_at: datetime,
) -> NewRun:
    return NewRun(
        response_id=response_id,
        owner_scope="integration-owner",
        model="background-report-agent",
        checkpoint_thread_id=f"background-integration:{response_id}",
        graph_version="background-report-v1",
        envelope={
            "model": "background-report-agent",
            "input": "Prepare a test report.",
            "background": True,
        },
        request_fingerprint="fixed-request-fingerprint",
        idempotency_digest=idempotency_digest,
        response={
            "id": response_id,
            "object": "response",
            "status": "queued",
            "output": [],
        },
        created_at=created_at,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def test_setup_is_concurrent_and_idempotent() -> None:
    async with (
        postgres_runtime(POSTGRES_URI) as runtime,
        runtime.pool.connection() as connection,
    ):
        await asyncio.gather(
            runtime.response_store.setup(),
            runtime.response_store.setup(),
        )
        cursor = await connection.execute(
            "SELECT data_type, character_maximum_length "
            "FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'lgos_background_responses' "
            "AND column_name = 'idempotency_digest'"
        )
        row = await cursor.fetchone()

    assert row == {
        "data_type": "character varying",
        "character_maximum_length": 64,
    }


async def _delete_runs(response_ids: set[str]) -> None:
    if not response_ids:
        return
    async with (
        postgres_runtime(POSTGRES_URI) as cleanup,
        cleanup.pool.connection() as connection,
    ):
        for response_id in response_ids:
            await connection.execute(
                "DELETE FROM lgos_background_responses WHERE response_id = %s",
                (response_id,),
            )


async def test_long_idempotency_key_is_hashed_before_persistence() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    response_ids: set[str] = set()
    long_key = "".join(uuid.uuid4().hex for _ in range(128))

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            graph = create_background_report_graph(runtime.checkpointer)
            graphs = GraphRegistry(
                registry={
                    "background-report-agent": create_background_report_config(
                        lambda: graph,
                        runtime.run_coordinator,
                    )
                }
            )
            backend = HatchetBackgroundBackend(
                workflow=Mock(
                    aio_run=AsyncMock(
                        return_value=SimpleNamespace(workflow_run_id="native-long-key")
                    )
                ),
                runs=Mock(),
                store=runtime.response_store,
            )
            request = ResponseCreateRequest(
                model="background-report-agent",
                input="Prepare a test report.",
                background=True,
                store=True,
            )

            first = await accept_background_response(
                request,
                graphs,
                backend,
                checkpoint_scope="integration-owner",
                idempotency_key=long_key,
            )
            response_ids.add(first.id)
            replay = await accept_background_response(
                request,
                graphs,
                backend,
                checkpoint_scope="integration-owner",
                idempotency_key=long_key,
            )
            stored = await runtime.response_store.get_internal(first.id)

            assert replay.id == first.id
            assert stored is not None
            assert stored.idempotency_digest is not None
            assert len(stored.idempotency_digest) == len(hashlib.sha256().hexdigest())
            assert long_key not in stored.model_dump_json()
    finally:
        await _delete_runs(response_ids)


async def test_pending_submissions_survive_restart_and_rotate():
    await setup_postgres_schema(POSTGRES_URI)
    now = datetime.now(UTC)
    runs = [
        _new_run(
            response_id=f"resp_{uuid.uuid4().hex}",
            idempotency_digest=_digest(str(uuid.uuid4())),
            created_at=now,
        )
        for _ in range(3)
    ]
    response_ids = {run.response_id for run in runs}
    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            for run in runs:
                await runtime.response_store.accept(run, capacity=10)
            await runtime.response_store.record_workflow_run(
                runs[0].response_id, "already-submitted", now=now
            )

        async with postgres_runtime(POSTGRES_URI) as restarted:
            first = await restarted.response_store.claim_pending_submissions(
                now=now + timedelta(seconds=1), limit=1
            )
            second = await restarted.response_store.claim_pending_submissions(
                now=now + timedelta(seconds=2), limit=1
            )
            assert len(first) == len(second) == 1
            assert {first[0].response_id, second[0].response_id} == {
                runs[1].response_id,
                runs[2].response_id,
            }
            for run in (first[0], second[0]):
                await restarted.response_store.record_workflow_run(
                    run.response_id,
                    f"native-{run.response_id}",
                    now=now,
                )
            assert (
                await restarted.response_store.claim_pending_submissions(
                    now=now, limit=10
                )
                == []
            )
    finally:
        await _delete_runs(response_ids)


async def test_store_survives_restart_with_native_receipt_and_cancellation() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    idempotency_digest = _digest(f"client-key/{uuid.uuid4().hex}")
    first = _new_run(
        response_id=f"resp_{uuid.uuid4().hex}",
        idempotency_digest=idempotency_digest,
        created_at=created_at,
    )
    second = _new_run(
        response_id=f"resp_{uuid.uuid4().hex}",
        idempotency_digest=idempotency_digest,
        created_at=created_at,
    )
    persisted_response_id: str | None = None

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            accepted = await asyncio.gather(
                runtime.response_store.accept(first, capacity=10),
                runtime.response_store.accept(second, capacity=10),
            )
            assert {item.response_id for item in accepted} == {accepted[0].response_id}
            persisted = accepted[0]
            persisted_response_id = persisted.response_id
            recorded = await runtime.response_store.record_workflow_run(
                persisted.response_id,
                "native-run",
                now=created_at + timedelta(seconds=1),
            )
            assert recorded is not None
            cancelled = await runtime.response_store.request_cancellation(
                persisted.response_id,
                persisted.owner_scope,
                {**persisted.response, "status": "cancelled"},
                now=created_at + timedelta(seconds=2),
                result_retention=timedelta(hours=1),
                idempotency_retention=timedelta(hours=24),
            )
            assert cancelled is not None
            assert cancelled.status is ResponseStatus.CANCELLED

        async with postgres_runtime(POSTGRES_URI) as restarted:
            recovered = await restarted.response_store.get_internal(
                persisted.response_id
            )
            assert recovered is not None
            assert recovered.status is ResponseStatus.CANCELLED
            assert recovered.workflow_run_id == "native-run"
            assert recovered.cancellation_pending
            assert recovered.cleanup_pending
            claimed = await restarted.response_store.claim_cancellations(
                now=created_at + timedelta(seconds=3),
                limit=1,
            )
            assert [run.response_id for run in claimed] == [persisted.response_id]
            assert await restarted.response_store.finish_cancellation(
                persisted.response_id,
                now=created_at + timedelta(seconds=4),
            )
    finally:
        await _delete_runs(
            {persisted_response_id} if persisted_response_id is not None else set()
        )


async def test_terminal_publication_does_not_overwrite_cancellation() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    run = _new_run(
        response_id=f"resp_{uuid.uuid4().hex}",
        idempotency_digest=_digest(str(uuid.uuid4())),
        created_at=created_at,
    )

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            accepted = await runtime.response_store.accept(run, capacity=10)
            in_progress = await runtime.response_store.mark_in_progress(
                run.response_id,
                now=created_at,
            )
            cancelled = await runtime.response_store.request_cancellation(
                run.response_id,
                run.owner_scope,
                {**run.response, "status": "cancelled"},
                now=created_at + timedelta(seconds=1),
                result_retention=timedelta(hours=1),
                idempotency_retention=timedelta(hours=24),
            )
            completed = await runtime.response_store.publish_terminal(
                run.response_id,
                {**run.response, "status": "completed"},
                now=created_at + timedelta(seconds=2),
                result_retention=timedelta(hours=1),
                idempotency_retention=timedelta(hours=24),
            )
            pending_receipts = await runtime.response_store.claim_pending_submissions(
                now=created_at + timedelta(seconds=3),
                limit=10,
            )

            assert accepted.response_id == run.response_id
            assert in_progress is not None
            assert in_progress.status is ResponseStatus.IN_PROGRESS
            assert cancelled is not None
            assert cancelled.status is ResponseStatus.CANCELLED
            assert cancelled.cancellation_pending
            assert completed is None
            assert [item.response_id for item in pending_receipts] == [run.response_id]
    finally:
        await _delete_runs({run.response_id})


async def test_expiry_batches_move_past_retained_tombstones() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime(2000, 1, 1, tzinfo=UTC)
    runs = [
        _new_run(
            response_id=f"resp_{uuid.uuid4().hex}",
            idempotency_digest=_digest(str(uuid.uuid4())),
            created_at=created_at + timedelta(microseconds=index),
        )
        for index in range(2)
    ]
    response_ids = {run.response_id for run in runs}

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            accepted = [
                await runtime.response_store.accept(run, capacity=10) for run in runs
            ]
            terminal_at = created_at + timedelta(seconds=1)
            for item in accepted:
                await runtime.response_store.publish_terminal(
                    item.response_id,
                    {**item.response, "status": "completed"},
                    now=terminal_at,
                    result_retention=timedelta(seconds=1),
                    idempotency_retention=timedelta(hours=1),
                )
                await runtime.response_store.finish_cleanup(
                    item.response_id,
                    now=terminal_at,
                )

            expired_at = terminal_at + timedelta(seconds=2)
            assert await runtime.response_store.expire(now=expired_at, limit=1) == 1
            assert await runtime.response_store.expire(now=expired_at, limit=1) == 1
            persisted = [
                await runtime.response_store.get_internal(run.response_id)
                for run in runs
            ]

            assert all(run is not None for run in persisted)
            assert all(run.response is None for run in persisted if run is not None)
            assert all(run.envelope == {} for run in persisted if run is not None)
    finally:
        await _delete_runs(response_ids)


async def test_cleanup_claims_rotate_past_an_unfinished_row() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    runs = [
        _new_run(
            response_id=f"resp_{uuid.uuid4().hex}",
            idempotency_digest=_digest(str(uuid.uuid4())),
            created_at=created_at,
        )
        for _ in range(2)
    ]
    response_ids = {run.response_id for run in runs}

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            for run in runs:
                await runtime.response_store.accept(run, capacity=10)
                await runtime.response_store.publish_terminal(
                    run.response_id,
                    {**run.response, "status": "completed"},
                    now=created_at,
                    result_retention=timedelta(hours=1),
                    idempotency_retention=timedelta(hours=1),
                )

            first = await runtime.response_store.claim_cleanup_ready(
                now=created_at + timedelta(seconds=1),
                limit=1,
            )
            second = await runtime.response_store.claim_cleanup_ready(
                now=created_at + timedelta(seconds=2),
                limit=1,
            )

            assert len(first) == len(second) == 1
            assert first[0].response_id != second[0].response_id
    finally:
        await _delete_runs(response_ids)
