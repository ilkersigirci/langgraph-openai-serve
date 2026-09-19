"""Restart and race coverage for the PostgreSQL background Response store."""

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from langgraph_openai_serve.background.store import (
    NewRun,
    ResponseStatus,
)

from lgos_demo_api.checkpointer import postgres_runtime, setup_postgres_schema

POSTGRES_URI = os.environ.get(
    "DEMO_API_TEST_POSTGRES_URI",
    "postgresql://lgos:lgos@localhost:3001/lgos",
)

pytestmark = pytest.mark.integration


def _new_run(*, response_id: str, run_key: str, created_at: datetime) -> NewRun:
    return NewRun(
        run_id=response_id,
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
        idempotency_key=run_key,
        response={
            "id": response_id,
            "object": "response",
            "status": "queued",
            "output": [],
        },
        created_at=created_at,
    )


async def _delete_runs(run_ids: set[str]) -> None:
    if not run_ids:
        return
    async with (
        postgres_runtime(POSTGRES_URI) as cleanup,
        cleanup.pool.connection() as connection,
    ):
        for run_id in run_ids:
            await connection.execute(
                "DELETE FROM lgos_background_responses WHERE run_id = %s",
                (run_id,),
            )


async def test_store_survives_restart_with_native_receipt_and_cancellation() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    run_key = str(uuid.uuid4())
    first = _new_run(
        response_id=f"resp_{uuid.uuid4().hex}",
        run_key=run_key,
        created_at=created_at,
    )
    second = _new_run(
        response_id=f"resp_{uuid.uuid4().hex}",
        run_key=run_key,
        created_at=created_at,
    )
    persisted_run_id: str | None = None

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            accepted = await asyncio.gather(
                runtime.response_store.accept(first, capacity=10),
                runtime.response_store.accept(second, capacity=10),
            )
            assert {item.run.response_id for item in accepted} == {
                accepted[0].run.response_id
            }
            assert sum(item.created for item in accepted) == 1
            persisted = accepted[0].run
            persisted_run_id = persisted.run_id
            recorded = await runtime.response_store.record_workflow_run(
                persisted.run_id,
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
            recovered = await restarted.response_store.get_internal(persisted.run_id)
            assert recovered is not None
            assert recovered.status is ResponseStatus.CANCELLED
            assert recovered.workflow_run_id == "native-run"
            assert recovered.cancellation_pending
            assert recovered.cleanup_pending
            claimed = await restarted.response_store.claim_cancellations(
                now=created_at + timedelta(seconds=3),
                limit=1,
            )
            assert [run.run_id for run in claimed] == [persisted.run_id]
            assert await restarted.response_store.finish_cancellation(
                persisted.run_id,
                now=created_at + timedelta(seconds=4),
            )
    finally:
        await _delete_runs(
            {persisted_run_id} if persisted_run_id is not None else set()
        )


async def test_terminal_publication_does_not_overwrite_cancellation() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    run = _new_run(
        response_id=f"resp_{uuid.uuid4().hex}",
        run_key=str(uuid.uuid4()),
        created_at=created_at,
    )

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            accepted = await runtime.response_store.accept(run, capacity=10)
            in_progress = await runtime.response_store.mark_in_progress(
                run.run_id,
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
                run.run_id,
                {**run.response, "status": "completed"},
                now=created_at + timedelta(seconds=2),
                result_retention=timedelta(hours=1),
                idempotency_retention=timedelta(hours=24),
            )

            assert accepted.created
            assert in_progress is not None
            assert in_progress.status is ResponseStatus.IN_PROGRESS
            assert cancelled is not None
            assert cancelled.status is ResponseStatus.CANCELLED
            assert completed is None
    finally:
        await _delete_runs({run.run_id})


async def test_expiry_batches_move_past_retained_tombstones() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime(2000, 1, 1, tzinfo=UTC)
    runs = [
        _new_run(
            response_id=f"resp_{uuid.uuid4().hex}",
            run_key=str(uuid.uuid4()),
            created_at=created_at + timedelta(microseconds=index),
        )
        for index in range(2)
    ]
    run_ids = {run.run_id for run in runs}

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            accepted = [
                await runtime.response_store.accept(run, capacity=10) for run in runs
            ]
            terminal_at = created_at + timedelta(seconds=1)
            for item in accepted:
                await runtime.response_store.publish_terminal(
                    item.run.run_id,
                    {**item.run.response, "status": "completed"},
                    now=terminal_at,
                    result_retention=timedelta(seconds=1),
                    idempotency_retention=timedelta(hours=1),
                )
                await runtime.response_store.finish_cleanup(
                    item.run.run_id,
                    now=terminal_at,
                )

            expired_at = terminal_at + timedelta(seconds=2)
            assert await runtime.response_store.expire(now=expired_at, limit=1) == 1
            assert await runtime.response_store.expire(now=expired_at, limit=1) == 1
            persisted = [
                await runtime.response_store.get_internal(run.run_id) for run in runs
            ]

            assert all(run is not None for run in persisted)
            assert all(run.response is None for run in persisted if run is not None)
            assert all(run.envelope == {} for run in persisted if run is not None)
    finally:
        await _delete_runs(run_ids)


async def test_cleanup_claims_rotate_past_an_unfinished_row() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    runs = [
        _new_run(
            response_id=f"resp_{uuid.uuid4().hex}",
            run_key=str(uuid.uuid4()),
            created_at=created_at,
        )
        for _ in range(2)
    ]
    run_ids = {run.run_id for run in runs}

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            for run in runs:
                await runtime.response_store.accept(run, capacity=10)
                await runtime.response_store.publish_terminal(
                    run.run_id,
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
            assert first[0].run_id != second[0].run_id
    finally:
        await _delete_runs(run_ids)
