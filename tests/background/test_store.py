"""Behavior tests for the orchestration-free Response store."""

from datetime import UTC, datetime, timedelta

import pytest

from langgraph_openai_serve import InMemoryResponseStore
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
    idempotency_key: str | None = None,
    fingerprint: str = "fingerprint",
) -> NewRun:
    return NewRun(
        run_id=f"run-{name}",
        response_id=f"resp-{name}",
        owner_scope="owner",
        model="model",
        checkpoint_thread_id=f"checkpoint-{name}",
        graph_version="v1",
        envelope={"model": "model", "input": "hello"},
        request_fingerprint=fingerprint,
        idempotency_key=idempotency_key,
        response={"id": f"resp-{name}", "status": "queued"},
        created_at=now,
    )


async def test_accept_enforces_capacity_and_idempotency() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    original = _new_run("one", now=now, idempotency_key="key")

    first = await store.accept(original, capacity=1)
    replay = await store.accept(
        _new_run("other", now=now, idempotency_key="key"),
        capacity=1,
    )

    assert first.created is True
    assert replay.created is False
    assert replay.run.run_id == original.run_id
    with pytest.raises(BackgroundCapacityError):
        await store.accept(_new_run("two", now=now), capacity=1)
    with pytest.raises(BackgroundIdempotencyConflictError):
        await store.accept(
            _new_run(
                "conflict",
                now=now,
                idempotency_key="key",
                fingerprint="different",
            ),
            capacity=2,
        )


async def test_workflow_receipt_is_idempotent_and_unsubmitted_rows_can_be_removed() -> (
    None
):
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=2)
    await store.accept(_new_run("two", now=now), capacity=2)

    recorded = await store.record_workflow_run(
        "run-one",
        "hatchet-one",
        now=now,
    )

    assert recorded is not None
    assert recorded.workflow_run_id == "hatchet-one"
    assert await store.record_workflow_run("run-one", "other", now=now) is None
    assert await store.discard_unsubmitted("run-one") is False
    assert await store.discard_unsubmitted("run-two") is True
    assert await store.get_internal("run-two") is None


async def test_cancellation_and_completion_have_one_terminal_winner() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=1)
    await store.record_workflow_run("run-one", "native-run", now=now)
    in_progress = await store.mark_in_progress("run-one", now=now)

    cancelled = await store.request_cancellation(
        "resp-one",
        "owner",
        {"id": "resp-one", "status": "cancelled"},
        now=now,
        result_retention=timedelta(hours=1),
        idempotency_retention=timedelta(hours=24),
    )
    completed = await store.publish_terminal(
        "run-one",
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


async def test_expiry_retains_then_removes_an_idempotency_tombstone() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    await store.accept(
        _new_run("one", now=now, idempotency_key="key"),
        capacity=1,
    )
    await store.publish_terminal(
        "run-one",
        {"id": "resp-one", "status": "completed"},
        now=now,
        result_retention=timedelta(seconds=1),
        idempotency_retention=timedelta(hours=1),
    )
    await store.finish_cleanup("run-one", now=now)

    assert await store.expire(now=now + timedelta(seconds=2), limit=10) == 1
    tombstone = await store.get_internal("run-one")
    assert tombstone is not None
    assert tombstone.response is None
    assert tombstone.envelope == {}
    with pytest.raises(BackgroundResponseExpiredError):
        await store.accept(
            _new_run(
                "replay",
                now=now + timedelta(seconds=2),
                idempotency_key="key",
            ),
            capacity=1,
        )

    assert await store.expire(now=now + timedelta(hours=2), limit=10) == 1
    assert await store.get_internal("run-one") is None


async def test_expiry_skips_retained_tombstones_without_starving_later_work() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    for name, key in (("one", "key"), ("two", None)):
        await store.accept(
            _new_run(name, now=now, idempotency_key=key),
            capacity=2,
        )
        await store.publish_terminal(
            f"run-{name}",
            {"id": f"resp-{name}", "status": "completed"},
            now=now,
            result_retention=timedelta(seconds=1),
            idempotency_retention=timedelta(hours=1),
        )
        await store.finish_cleanup(f"run-{name}", now=now)

    expired_at = now + timedelta(seconds=2)
    assert await store.expire(now=expired_at, limit=1) == 1
    assert await store.expire(now=expired_at, limit=1) == 1
    assert await store.get_internal("run-two") is None


async def test_cleanup_claim_rotates_past_an_unfinished_row() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    for name in ("one", "two"):
        await store.accept(_new_run(name, now=now), capacity=2)
        await store.publish_terminal(
            f"run-{name}",
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
    assert first[0].run_id != second[0].run_id


async def test_expiry_retains_a_pending_native_cancellation() -> None:
    store = InMemoryResponseStore()
    now = datetime.now(UTC)
    await store.accept(_new_run("one", now=now), capacity=1)
    await store.record_workflow_run("run-one", "native-run", now=now)
    await store.request_cancellation(
        "resp-one",
        "owner",
        {"id": "resp-one", "status": "cancelled"},
        now=now,
        result_retention=timedelta(seconds=1),
        idempotency_retention=timedelta(seconds=1),
    )
    await store.finish_cleanup("run-one", now=now)
    expired_at = now + timedelta(seconds=2)

    assert await store.expire(now=expired_at, limit=10) == 1
    assert await store.expire(now=expired_at, limit=10) == 0
    retained = await store.get_internal("run-one")
    assert retained is not None
    assert retained.response is None
    assert retained.cancellation_pending is True

    assert await store.finish_cancellation("run-one", now=expired_at) is True
    assert await store.expire(now=expired_at, limit=10) == 1
    assert await store.get_internal("run-one") is None
