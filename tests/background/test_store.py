"""Behavior tests for the orchestration-free Response store."""

from datetime import UTC, datetime, timedelta

import pytest
from anyio import Event, create_task_group, fail_after

from langgraph_openai_serve import ResponseStore
from langgraph_openai_serve.background.store import (
    BackgroundCapacityError,
    BackgroundIdempotencyConflictError,
    BackgroundResponseExpiredError,
    NewRun,
    ResponseStatus,
)


def _new_run(
    name: str,
    *,
    now: datetime,
    idempotency_digest: str | None = None,
    fingerprint: str = "fingerprint",
) -> NewRun:
    return NewRun(
        response_id=f"resp-{name}",
        owner_scope="owner",
        model="model",
        checkpoint_thread_id=f"checkpoint-{name}",
        graph_version="v1",
        envelope={"model": "model", "input": "hello"},
        request_fingerprint=fingerprint,
        idempotency_digest=idempotency_digest,
        response={"id": f"resp-{name}", "status": "queued"},
        created_at=now,
    )


async def test_accept_enforces_capacity_and_idempotency(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    original = _new_run("one", now=now, idempotency_digest="digest")

    first = await store.accept(original, capacity=1)
    replay = await store.accept(
        _new_run("other", now=now, idempotency_digest="digest"),
        capacity=1,
    )

    assert first.response_id == original.response_id
    assert replay.response_id == original.response_id
    with pytest.raises(BackgroundCapacityError):
        await store.accept(_new_run("two", now=now), capacity=1)
    with pytest.raises(BackgroundIdempotencyConflictError):
        await store.accept(
            _new_run(
                "conflict",
                now=now,
                idempotency_digest="digest",
                fingerprint="different",
            ),
            capacity=2,
        )


async def test_workflow_receipt_is_idempotent(response_store: ResponseStore) -> None:
    store = response_store
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=2)

    recorded = await store.record_workflow_run(
        "resp-one",
        "workflow-one",
        now=now,
    )

    assert recorded is not None
    assert recorded.workflow_run_id == "workflow-one"
    assert await store.record_workflow_run("resp-one", "other", now=now) is None


async def test_pending_submission_claim_skips_submitted_and_terminal_runs(
    response_store: ResponseStore,
):
    store = response_store
    now = datetime.now(UTC)
    for name in ("pending", "submitted", "terminal"):
        await store.accept(_new_run(name, now=now), capacity=3)
    await store.record_workflow_run("resp-submitted", "native-run", now=now)
    await store.publish_terminal(
        "resp-terminal",
        {"id": "resp-terminal", "status": "completed"},
        now=now,
        result_retention=timedelta(hours=1),
        idempotency_retention=timedelta(hours=1),
    )

    pending = await store.claim_pending_submissions(now=now, limit=10)

    assert [run.response_id for run in pending] == ["resp-pending"]


async def test_cancellation_and_completion_have_one_terminal_winner(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=1)
    await store.record_workflow_run("resp-one", "native-run", now=now)
    in_progress = await store.mark_in_progress("resp-one", now=now)

    cancelled = await store.request_cancellation(
        "resp-one",
        "owner",
        {"id": "resp-one", "status": "cancelled"},
        now=now,
        result_retention=timedelta(hours=1),
        idempotency_retention=timedelta(hours=24),
    )
    completed = await store.publish_terminal(
        "resp-one",
        {"id": "resp-one", "status": "completed"},
        now=now,
        result_retention=timedelta(hours=1),
        idempotency_retention=timedelta(hours=24),
    )

    assert in_progress is not None
    assert in_progress.status is ResponseStatus.IN_PROGRESS
    assert cancelled is not None
    assert cancelled.status is ResponseStatus.CANCELLED
    assert cancelled.cancellation_pending is True
    assert cancelled.cleanup_pending is True
    assert completed is None


async def test_cancelled_run_without_receipt_remains_recoverable(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=1)

    cancelled = await store.request_cancellation(
        "resp-one",
        "owner",
        {"id": "resp-one", "status": "cancelled"},
        now=now,
        result_retention=timedelta(seconds=1),
        idempotency_retention=timedelta(seconds=1),
    )
    pending = await store.claim_pending_submissions(now=now, limit=1)
    recorded = await store.record_workflow_run(
        "resp-one",
        "native-run",
        now=now,
    )
    cancellations = await store.claim_cancellations(now=now, limit=1)

    assert cancelled is not None
    assert cancelled.cancellation_pending is True
    assert [run.response_id for run in pending] == ["resp-one"]
    assert recorded is not None
    assert recorded.cancellation_pending is True
    assert [run.response_id for run in cancellations] == ["resp-one"]


async def test_expiry_retains_then_removes_an_idempotency_tombstone(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    await store.accept(
        _new_run("one", now=now, idempotency_digest="digest"),
        capacity=1,
    )
    await store.publish_terminal(
        "resp-one",
        {"id": "resp-one", "status": "completed"},
        now=now,
        result_retention=timedelta(seconds=1),
        idempotency_retention=timedelta(hours=1),
    )
    await store.finish_cleanup("resp-one", now=now)

    assert await store.expire(now=now + timedelta(seconds=2), limit=10) == 1
    tombstone = await store.get_internal("resp-one")
    assert tombstone is not None
    assert tombstone.response is None
    assert tombstone.envelope == {}
    with pytest.raises(BackgroundResponseExpiredError):
        await store.accept(
            _new_run(
                "replay",
                now=now + timedelta(seconds=2),
                idempotency_digest="digest",
            ),
            capacity=1,
        )

    assert await store.expire(now=now + timedelta(hours=2), limit=10) == 1
    assert await store.get_internal("resp-one") is None


@pytest.mark.parametrize("same_key", [False, True], ids=["capacity", "idempotency"])
async def test_concurrent_acceptance_is_atomic(response_store_pair, same_key):
    now = datetime.now(UTC)
    start = Event()
    results = []
    rejected = []

    async def accept(store, name):
        await start.wait()
        try:
            results.append(
                await store.accept(
                    _new_run(
                        name,
                        now=now,
                        idempotency_digest="shared" if same_key else None,
                    ),
                    capacity=1,
                )
            )
        except BackgroundCapacityError:
            rejected.append(name)

    with fail_after(5):
        async with create_task_group() as tasks:
            for index, store in enumerate(response_store_pair):
                tasks.start_soon(accept, store, str(index))
            start.set()

    assert len({run.response_id for run in results}) == 1
    assert len(results) == (2 if same_key else 1)
    assert len(rejected) == (0 if same_key else 1)
    for store in response_store_pair:
        assert await store.get_internal(results[0].response_id) == results[0]


async def test_concurrent_terminal_transitions_keep_one_winner(response_store_pair):
    writer, peer = response_store_pair
    now = datetime.now(UTC)
    await writer.accept(_new_run("race", now=now), capacity=1)
    start = Event()

    async def complete():
        await start.wait()
        await writer.publish_terminal(
            "resp-race",
            {"id": "resp-race", "status": "completed"},
            now=now,
            result_retention=timedelta(hours=1),
            idempotency_retention=timedelta(hours=1),
        )

    async def cancel():
        await start.wait()
        await peer.request_cancellation(
            "resp-race",
            "owner",
            {"id": "resp-race", "status": "cancelled"},
            now=now,
            result_retention=timedelta(hours=1),
            idempotency_retention=timedelta(hours=1),
        )

    with fail_after(5):
        async with create_task_group() as tasks:
            tasks.start_soon(complete)
            tasks.start_soon(cancel)
            start.set()

    winner = await writer.get_internal("resp-race")
    assert winner is not None
    assert winner.status in {ResponseStatus.COMPLETED, ResponseStatus.CANCELLED}
    assert winner.cleanup_pending
    assert winner.cancellation_pending == (winner.status is ResponseStatus.CANCELLED)
    await complete()
    await cancel()
    assert await peer.get_internal("resp-race") == winner


async def test_public_access_enforces_owner_and_expiry_before_cleanup(response_store):
    now = datetime.now(UTC)
    await response_store.accept(_new_run("private", now=now), capacity=1)
    assert await response_store.get("resp-private", "other", now=now) is None
    assert (
        await response_store.request_cancellation(
            "resp-private",
            "other",
            {"id": "resp-private", "status": "cancelled"},
            now=now,
            result_retention=timedelta(seconds=1),
            idempotency_retention=timedelta(seconds=1),
        )
        is None
    )
    await response_store.publish_terminal(
        "resp-private",
        {"id": "resp-private", "status": "completed"},
        now=now,
        result_retention=timedelta(seconds=1),
        idempotency_retention=timedelta(seconds=1),
    )
    assert await response_store.get("resp-private", "owner", now=now) is not None
    assert (
        await response_store.get(
            "resp-private", "owner", now=now + timedelta(seconds=1)
        )
        is None
    )
    assert await response_store.get_internal("resp-private") is not None


async def test_expiry_skips_retained_tombstones_without_starving_later_work(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    for name, key in (("one", "key"), ("two", None)):
        await store.accept(
            _new_run(name, now=now, idempotency_digest=key),
            capacity=2,
        )
        await store.publish_terminal(
            f"resp-{name}",
            {"id": f"resp-{name}", "status": "completed"},
            now=now,
            result_retention=timedelta(seconds=1),
            idempotency_retention=timedelta(hours=1),
        )
        await store.finish_cleanup(f"resp-{name}", now=now)

    expired_at = now + timedelta(seconds=2)
    assert await store.expire(now=expired_at, limit=1) == 1
    assert await store.expire(now=expired_at, limit=1) == 1
    assert await store.get_internal("resp-two") is None


async def test_cleanup_claim_rotates_past_an_unfinished_row(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    for name in ("one", "two"):
        await store.accept(_new_run(name, now=now), capacity=2)
        await store.publish_terminal(
            f"resp-{name}",
            {"id": f"resp-{name}", "status": "completed"},
            now=now,
            result_retention=timedelta(hours=1),
            idempotency_retention=timedelta(hours=24),
        )

    first = await store.claim_cleanup_ready(
        now=now + timedelta(seconds=1),
        limit=1,
    )
    second = await store.claim_cleanup_ready(
        now=now + timedelta(seconds=2),
        limit=1,
    )

    assert len(first) == len(second) == 1
    assert first[0].response_id != second[0].response_id


async def test_expiry_retains_a_pending_native_cancellation(
    response_store: ResponseStore,
) -> None:
    store = response_store
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=1)
    await store.record_workflow_run("resp-one", "native-run", now=now)
    await store.request_cancellation(
        "resp-one",
        "owner",
        {"id": "resp-one", "status": "cancelled"},
        now=now,
        result_retention=timedelta(seconds=1),
        idempotency_retention=timedelta(seconds=1),
    )
    await store.finish_cleanup("resp-one", now=now)
    expired_at = now + timedelta(seconds=2)

    assert await store.expire(now=expired_at, limit=10) == 1
    assert await store.expire(now=expired_at, limit=10) == 0
    retained = await store.get_internal("resp-one")
    assert retained is not None
    assert retained.response is None
    assert retained.cancellation_pending is True

    assert await store.finish_cancellation("resp-one", now=expired_at) is True
    assert await store.expire(now=expired_at, limit=10) == 1
    assert await store.get_internal("resp-one") is None
