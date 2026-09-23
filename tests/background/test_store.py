"""Behavior tests for the orchestration-free Response store."""

from datetime import UTC, datetime, timedelta

from anyio import Event, create_task_group, fail_after

from langgraph_openai_serve import ResponseStore
from langgraph_openai_serve.background.store import NewRun, ResponseStatus


def _new_run(name: str, *, now: datetime, digest: str | None = None) -> NewRun:
    return NewRun(
        response_id=f"resp-{name}",
        owner_scope="owner",
        model="model",
        checkpoint_thread_id=f"checkpoint-{name}",
        graph_version="v1",
        envelope={"model": "model", "input": "hello"},
        response={"id": f"resp-{name}", "status": "queued"},
        created_at=now,
        initial_call_ids=("call-1",),
        idempotency_digest=digest,
        request_fingerprint="fingerprint" if digest else None,
    )


async def _finish(
    store: ResponseStore,
    name: str,
    status: str,
    *,
    now: datetime,
    retention: timedelta = timedelta(hours=1),
):
    return await store.finish(
        f"resp-{name}",
        {"id": f"resp-{name}", "status": status},
        now=now,
        result_retention=retention,
    )


async def test_created_run_round_trips(response_store: ResponseStore) -> None:
    now = datetime.now(UTC)
    new_run = _new_run("one", now=now)

    created = await response_store.create(new_run)

    assert created.status is ResponseStatus.QUEUED
    assert created.model_dump(include=set(NewRun.model_fields)) == new_run.model_dump()
    assert await response_store.get("resp-one") == created
    assert await response_store.get("resp-missing") is None


async def test_concurrent_creates_with_one_digest_keep_one_run(
    response_store_pair: tuple[ResponseStore, ResponseStore],
) -> None:
    now = datetime.now(UTC)
    start = Event()
    created = []

    async def create(store: ResponseStore, name: str) -> None:
        await start.wait()
        created.append(await store.create(_new_run(name, now=now, digest="key")))

    with fail_after(5):
        async with create_task_group() as tasks:
            for index, store in enumerate(response_store_pair):
                tasks.start_soon(create, store, f"try-{index}")
            start.set()

    assert created[0] == created[1]
    other = await response_store_pair[0].create(_new_run("other", now=now))
    assert other.response_id == "resp-other"


async def test_expiry_releases_the_idempotency_digest(
    response_store: ResponseStore,
) -> None:
    now = datetime.now(UTC)
    await response_store.create(_new_run("first", now=now, digest="key"))
    await _finish(
        response_store, "first", "completed", now=now, retention=timedelta(seconds=1)
    )
    await response_store.finish_cleanup("resp-first", now=now)
    await response_store.expire(now=now + timedelta(seconds=2), limit=10)

    second = await response_store.create(_new_run("second", now=now, digest="key"))

    assert second.response_id == "resp-second"


async def test_in_progress_updates_the_public_snapshot_until_terminal(
    response_store: ResponseStore,
) -> None:
    now = datetime.now(UTC)
    await response_store.create(_new_run("one", now=now))

    in_progress = await response_store.mark_in_progress("resp-one", now=now)
    await _finish(response_store, "one", "completed", now=now)

    assert in_progress is not None
    assert in_progress.status is ResponseStatus.IN_PROGRESS
    assert in_progress.response["status"] == "in_progress"
    assert await response_store.mark_in_progress("resp-one", now=now) is None


async def test_claim_queued_rotates_through_old_queued_runs_only(
    response_store: ResponseStore,
) -> None:
    now = datetime.now(UTC)
    for name in ("old", "older", "started"):
        await response_store.create(_new_run(name, now=now - timedelta(hours=1)))
    await response_store.create(_new_run("recent", now=now))
    await response_store.mark_in_progress("resp-started", now=now)

    first = await response_store.claim_queued(
        created_before=now, now=now + timedelta(seconds=1), limit=1
    )
    second = await response_store.claim_queued(
        created_before=now, now=now + timedelta(seconds=2), limit=1
    )
    everything = await response_store.claim_queued(
        created_before=now, now=now + timedelta(seconds=3), limit=10
    )

    assert {first[0].response_id, second[0].response_id} == {"resp-old", "resp-older"}
    assert {run.response_id for run in everything} == {"resp-old", "resp-older"}


async def test_first_terminal_outcome_wins(response_store: ResponseStore) -> None:
    now = datetime.now(UTC)
    await response_store.create(_new_run("one", now=now))

    cancelled = await _finish(response_store, "one", "cancelled", now=now)
    late = await _finish(
        response_store,
        "one",
        "completed",
        now=now + timedelta(seconds=1),
    )

    assert cancelled is not None
    assert cancelled.status is ResponseStatus.CANCELLED
    assert cancelled.cleanup_pending
    assert cancelled.result_expires_at == now + timedelta(hours=1)
    assert late == cancelled
    assert await _finish(response_store, "missing", "completed", now=now) is None


async def test_concurrent_terminal_transitions_keep_one_winner(
    response_store_pair: tuple[ResponseStore, ResponseStore],
) -> None:
    writer, peer = response_store_pair
    now = datetime.now(UTC)
    await writer.create(_new_run("race", now=now))
    start = Event()
    winners = []

    async def finish(store: ResponseStore, status: str) -> None:
        await start.wait()
        winners.append(await _finish(store, "race", status, now=now))

    with fail_after(5):
        async with create_task_group() as tasks:
            tasks.start_soon(finish, writer, "completed")
            tasks.start_soon(finish, peer, "cancelled")
            start.set()

    assert winners[0] is not None
    assert winners[0].status in {ResponseStatus.COMPLETED, ResponseStatus.CANCELLED}
    assert winners[1] == winners[0]
    assert await peer.get("resp-race") == winners[0]


async def test_cleanup_claim_rotates_past_an_unfinished_row(
    response_store: ResponseStore,
) -> None:
    now = datetime.now(UTC)
    for name in ("one", "two", "active"):
        await response_store.create(_new_run(name, now=now))
    for name in ("one", "two"):
        await _finish(response_store, name, "completed", now=now)

    first = await response_store.claim_cleanup_ready(
        now=now + timedelta(seconds=1),
        limit=1,
    )
    second = await response_store.claim_cleanup_ready(
        now=now + timedelta(seconds=2),
        limit=1,
    )
    await response_store.finish_cleanup("resp-one", now=now)
    await response_store.finish_cleanup("resp-two", now=now)

    assert len(first) == len(second) == 1
    assert {first[0].response_id, second[0].response_id} == {"resp-one", "resp-two"}
    assert await response_store.claim_cleanup_ready(now=now, limit=10) == []


async def test_expiry_waits_for_checkpoint_cleanup_and_respects_limit(
    response_store: ResponseStore,
) -> None:
    now = datetime.now(UTC)
    for name in ("one", "two", "pending", "active"):
        await response_store.create(_new_run(name, now=now))
    for name in ("one", "two", "pending"):
        await _finish(
            response_store,
            name,
            "completed",
            now=now,
            retention=timedelta(seconds=1),
        )
    for name in ("one", "two"):
        await response_store.finish_cleanup(f"resp-{name}", now=now)

    expired_at = now + timedelta(seconds=2)
    assert await response_store.expire(now=now, limit=10) == 0
    assert await response_store.expire(now=expired_at, limit=1) == 1
    assert await response_store.expire(now=expired_at, limit=10) == 1

    assert await response_store.get("resp-one") is None
    assert await response_store.get("resp-two") is None
    assert await response_store.get("resp-pending") is not None
    assert await response_store.get("resp-active") is not None
