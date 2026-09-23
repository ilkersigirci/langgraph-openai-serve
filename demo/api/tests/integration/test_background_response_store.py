"""Restart and race coverage for the PostgreSQL background Response store."""

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from langgraph_openai_serve.background.store import NewRun, ResponseStatus
from langgraph_openai_serve.integrations.background.postgres import MIGRATIONS

from lgos_demo_api.persistence.postgres import postgres_runtime, setup_postgres_schema

POSTGRES_URI = os.environ.get(
    "DEMO_API_TEST_POSTGRES_URI",
    "postgresql://lgos:lgos@localhost:3001/lgos",
)

pytestmark = pytest.mark.integration


def _new_run(*, created_at: datetime) -> NewRun:
    response_id = f"resp_{uuid.uuid4().hex}"
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
        response={
            "id": response_id,
            "object": "response",
            "status": "queued",
            "output": [],
        },
        created_at=created_at,
        initial_call_ids=("call-1",),
    )


async def _delete_runs(response_ids: set[str]) -> None:
    async with (
        postgres_runtime(POSTGRES_URI) as cleanup,
        cleanup.pool.connection() as connection,
    ):
        await connection.execute(
            "DELETE FROM lgos_background_responses WHERE response_id = ANY(%s)",
            (list(response_ids),),
        )


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
            "SELECT count(*) AS applied, max(v) AS latest "
            "FROM lgos_background_migrations"
        )
        row = await cursor.fetchone()

    # Version 0 creates the migrations table itself and is never recorded.
    assert row == {"applied": len(MIGRATIONS) - 1, "latest": len(MIGRATIONS) - 1}


async def test_run_survives_restart_and_keeps_one_terminal_winner() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    run = _new_run(created_at=created_at)

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            created = await runtime.response_store.create(run)
            in_progress = await runtime.response_store.mark_in_progress(
                run.response_id,
                now=created_at,
            )

        async with postgres_runtime(POSTGRES_URI) as restarted:
            store = restarted.response_store
            winners = await asyncio.gather(
                store.finish(
                    run.response_id,
                    {**run.response, "status": "cancelled"},
                    now=created_at + timedelta(seconds=1),
                    result_retention=timedelta(hours=1),
                ),
                store.finish(
                    run.response_id,
                    {**run.response, "status": "completed"},
                    now=created_at + timedelta(seconds=1),
                    result_retention=timedelta(hours=1),
                ),
            )
            recovered = await store.get(run.response_id)

        assert created.model_dump(include=set(NewRun.model_fields)) == run.model_dump()
        assert in_progress is not None
        assert in_progress.response["status"] == "in_progress"
        assert winners[0] is not None
        assert winners[0] == winners[1] == recovered
        assert winners[0].status in {
            ResponseStatus.CANCELLED,
            ResponseStatus.COMPLETED,
        }
        assert winners[0].response["status"] == winners[0].status.value
        assert winners[0].cleanup_pending
    finally:
        await _delete_runs({run.response_id})


async def test_cleanup_rotates_and_expiry_waits_for_cleanup() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime(2000, 1, 1, tzinfo=UTC)
    runs = [_new_run(created_at=created_at) for _ in range(2)]
    response_ids = {run.response_id for run in runs}

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            store = runtime.response_store
            for run in runs:
                await store.create(run)
                await store.finish(
                    run.response_id,
                    {**run.response, "status": "completed"},
                    now=created_at,
                    result_retention=timedelta(seconds=1),
                )

            first = await store.claim_cleanup_ready(
                now=created_at + timedelta(seconds=1),
                limit=1,
            )
            second = await store.claim_cleanup_ready(
                now=created_at + timedelta(seconds=2),
                limit=1,
            )
            expired_at = created_at + timedelta(seconds=3)
            expired_before_cleanup = await store.expire(now=expired_at, limit=10)
            for run in runs:
                await store.finish_cleanup(run.response_id, now=expired_at)
            expired = await store.expire(now=expired_at, limit=10)

        assert len(first) == len(second) == 1
        assert {first[0].response_id, second[0].response_id} == response_ids
        assert expired_before_cleanup == 0
        assert expired == len(runs)
    finally:
        await _delete_runs(response_ids)


async def test_concurrent_creates_with_one_digest_keep_one_run() -> None:
    await setup_postgres_schema(POSTGRES_URI)
    created_at = datetime.now(UTC)
    digest = uuid.uuid4().hex
    runs = [
        _new_run(created_at=created_at).model_copy(
            update={"idempotency_digest": digest, "request_fingerprint": "same"}
        )
        for _ in range(2)
    ]

    try:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            created = await asyncio.gather(
                *(runtime.response_store.create(run) for run in runs)
            )

        assert created[0] == created[1]
        assert created[0].response_id in {run.response_id for run in runs}
    finally:
        await _delete_runs({run.response_id for run in runs})
