from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from anyio import CancelScope, Event, create_task_group, fail_after
from anyio.lowlevel import checkpoint
from fastapi import FastAPI, status
from httpx2 import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from openai import (
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)

from langgraph_openai_serve import (
    BackgroundSettings,
    BackgroundWorker,
    ClientSettings,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    InMemoryBackgroundBackend,
    InMemoryResponseStore,
    LanggraphOpenaiServe,
    NewRun,
    RunCoordinator,
    RunLease,
    RunLeaseLostError,
    StoredRun,
)
from langgraph_openai_serve.graph.coordination import InMemoryRunCoordinator
from langgraph_openai_serve.graph.graph_registry import GraphConfigurationError
from tests.background.fakes import MemoryBackgroundBackend
from tests.graph.support.schemas import MessageState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import Request
    from openai.types.responses import Response


@dataclass
class _Environment:
    client: AsyncOpenAI
    http: AsyncClient
    backend: MemoryBackgroundBackend
    worker: BackgroundWorker
    store: InMemoryResponseStore
    invocations: list[str]


@asynccontextmanager
async def _environment(  # ruff: ignore[too-many-arguments] - Test fixture options each isolate one boundary.
    *,
    owner: str = "tenant-a",
    runtime_enabled: bool = True,
    model_enabled: bool = True,
    settings: BackgroundSettings | None = None,
    fail_after_checkpoint: int = 0,
    execution_gate: tuple[Event, Event] | None = None,
    render_error: Exception | None = None,
    coordinator: RunCoordinator | None = None,
    response_store: InMemoryResponseStore | None = None,
) -> AsyncIterator[_Environment]:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        model = FakeListChatModel(responses=["background hello"])
        invocations: list[str] = []

        async def generate(state: MessageState):
            invocations.append(str(state["messages"][-1].text))
            if execution_gate is not None:
                started, allow_execution = execution_gate
                started.set()
                await allow_execution.wait()
            return {"messages": [await model.ainvoke(state["messages"])]}

        failures_remaining = fail_after_checkpoint

        async def finalize(_state: MessageState):
            nonlocal failures_remaining
            if failures_remaining:
                failures_remaining -= 1
                message = "injected graph failure"
                raise OSError(message)
            return {}

        workflow = StateGraph(MessageState).add_node("generate", generate)
        workflow.set_entry_point("generate")
        if fail_after_checkpoint:
            workflow.add_node("finalize", finalize)
            workflow.add_edge("generate", "finalize")
            workflow.set_finish_point("finalize")
        else:
            workflow.set_finish_point("generate")
        graph = workflow.compile(checkpointer=checkpointer)

        def render_output(_output: object):
            assert render_error is not None
            raise render_error

        registry = GraphRegistry(
            registry={
                "background": GraphConfig(
                    graph=graph,
                    description="Background test graph",
                    features=(
                        frozenset({GraphFeature.BACKGROUND})
                        if model_enabled
                        else frozenset()
                    ),
                    run_coordinator=(
                        (coordinator or InMemoryRunCoordinator())
                        if model_enabled
                        else None
                    ),
                    output_to_message=render_output if render_error else None,
                )
            }
        )
        store = response_store or InMemoryResponseStore()
        backend = MemoryBackgroundBackend(store=store, settings=settings)
        worker = BackgroundWorker(graphs=registry, store=store, settings=settings)

        def checkpoint_scope(request: Request) -> str:
            return request.headers.get("x-owner", owner)

        app = (
            LanggraphOpenaiServe(
                graphs=registry,
                checkpoint_scope=checkpoint_scope,
                background=backend if runtime_enabled else None,
            )
            .bind_openai_api()
            .app
        )
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test") as http,
            AsyncOpenAI(
                api_key="test",
                base_url="http://test/v1",
                http_client=http,
                max_retries=0,
            ) as client,
        ):
            yield _Environment(client, http, backend, worker, store, invocations)


async def _execute_next(environment: _Environment):
    job = await environment.backend.receive()
    assert job is not None
    await environment.worker.execute(job)
    return job


@asynccontextmanager
async def _in_memory_client(
    environment: _Environment,
    *,
    settings: BackgroundSettings | None = None,
    maintenance_interval: timedelta = timedelta(minutes=1),
) -> AsyncIterator[tuple[AsyncOpenAI, InMemoryBackgroundBackend]]:
    backend = InMemoryBackgroundBackend(
        graphs=environment.worker.graphs,
        settings=settings,
        maintenance_interval=maintenance_interval,
    )
    app = FastAPI(lifespan=backend.lifespan)
    LanggraphOpenaiServe(
        app=app,
        graphs=environment.worker.graphs,
        background=backend,
    ).bind_openai_api()
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as http,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http,
            max_retries=0,
        ) as client,
    ):
        yield client, backend


async def _terminal_response(client: AsyncOpenAI, response_id: str) -> Response:
    with fail_after(2):
        while True:
            response = await client.responses.retrieve(response_id)
            if response.status not in {"queued", "in_progress"}:
                return response
            await checkpoint()


async def test_background_response_is_polled_and_executed_outside_post() -> None:
    async with _environment() as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )

        assert created.status == "queued"
        assert created.background is True
        assert created.created_at.is_integer()
        assert created.output == []
        assert environment.invocations == []
        assert await environment.client.responses.retrieve(created.id) == created

        job = await _execute_next(environment)
        completed = await environment.client.responses.retrieve(created.id)

        assert completed.status == "completed"
        assert completed.output_text == "background hello"
        assert environment.invocations == ["Hello"]

        await environment.worker.execute(job)
        assert environment.invocations == ["Hello"]


async def test_in_memory_backend_executes_without_an_external_engine() -> None:
    async with _environment() as environment:
        async with _in_memory_client(environment) as (client, _backend):
            created = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )
            completed = await _terminal_response(client, created.id)

        assert completed.status == "completed"
        assert completed.output_text == "background hello"


async def test_in_memory_backend_finalizes_execution_failure() -> None:
    async with _environment(fail_after_checkpoint=1) as environment:
        async with _in_memory_client(environment) as (client, _backend):
            created = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )
            failed = await _terminal_response(client, created.id)

        assert failed.status == "failed"
        assert failed.error is not None


async def test_in_memory_backend_expires_results_and_checkpoints() -> None:
    settings = BackgroundSettings(result_retention=timedelta(milliseconds=50))
    async with _environment() as environment:
        async with _in_memory_client(
            environment,
            settings=settings,
            maintenance_interval=timedelta(milliseconds=10),
        ) as (client, backend):
            created = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )
            completed = await _terminal_response(client, created.id)
            with fail_after(2):
                while await backend.store.get(created.id) is not None:
                    await checkpoint()
            with pytest.raises(NotFoundError):
                await client.responses.retrieve(created.id)

        assert completed.status == "completed"


async def test_in_memory_backend_starts_a_run_whose_request_was_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _environment() as environment:
        async with _in_memory_client(environment) as (client, backend):
            create_started = Event()
            release_create = Event()
            response_ids: list[str] = []
            store_create = backend.store.create

            async def delayed_create(run: NewRun) -> StoredRun:
                response_ids.append(run.response_id)
                create_started.set()
                await release_create.wait()
                return await store_create(run)

            monkeypatch.setattr(backend.store, "create", delayed_create)
            request_scope = CancelScope()

            async def create() -> None:
                with request_scope:
                    await client.responses.create(
                        model="background",
                        input="Hello",
                        background=True,
                    )

            with fail_after(2):
                async with create_task_group() as tasks:
                    tasks.start_soon(create)
                    await create_started.wait()
                    request_scope.cancel()
                    await checkpoint()
                    release_create.set()

            assert len(response_ids) == 1
            completed = await _terminal_response(client, response_ids[0])

        assert completed.status == "completed"


async def test_background_capable_model_preserves_foreground_execution() -> None:
    async with _environment() as environment:
        response = await environment.client.responses.create(
            model="background",
            input="Foreground",
            background=False,
        )

        assert response.status == "completed"
        assert response.output_text == "background hello"


async def test_background_requires_a_backend_and_an_opted_in_model() -> None:
    async with _environment(runtime_enabled=False) as environment:
        with pytest.raises(BadRequestError, match="not configured"):
            await environment.client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )

    async with _environment(model_enabled=False) as environment:
        with pytest.raises(BadRequestError, match="does not support"):
            await environment.client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )


@pytest.mark.parametrize("encoded", ['{"count":"invalid"}', "not-json"])
async def test_invalid_background_settings_are_rejected_before_persistence(encoded):
    class Settings(ClientSettings):
        count: int = 1

    async with _environment() as environment:
        config = environment.worker.graphs.get_graph("background")
        environment.worker.graphs.register(
            "background",
            config.model_copy(update={"client_settings": Settings}),
        )
        with pytest.raises(BadRequestError) as error:
            await environment.client.responses.create(
                model="background",
                input="Hello",
                background=True,
                metadata={"lgos_settings": encoded},
            )

        assert error.value.param == "metadata.lgos_settings"
        assert await environment.backend.receive() is None
        assert environment.invocations == []


async def test_streaming_background_create_and_retrieval_are_rejected() -> None:
    async with _environment() as environment:
        response = await environment.http.post(
            "/v1/responses",
            json={
                "model": "background",
                "input": "Hello",
                "background": True,
                "stream": True,
            },
        )
        retrieval = await environment.http.get(
            "/v1/responses/resp_unknown",
            params={"stream": "true"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["error"]["param"] == "stream"
        assert retrieval.status_code == status.HTTP_400_BAD_REQUEST
        assert retrieval.json()["error"]["param"] == "stream"


async def test_retrieval_and_cancellation_are_owner_scoped() -> None:
    async with _environment() as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )

        hidden = await environment.http.get(
            f"/v1/responses/{created.id}",
            headers={"x-owner": "tenant-b"},
        )
        cancelled = await environment.client.responses.cancel(created.id)
        repeated = await environment.client.responses.cancel(created.id)

        assert hidden.status_code == status.HTTP_404_NOT_FOUND
        assert cancelled.status == "cancelled"
        assert repeated == cancelled
        assert await environment.backend.receive() is None


async def test_maintenance_resubmits_a_run_whose_submission_failed() -> None:
    settings = BackgroundSettings(resubmit_after=timedelta(microseconds=1))
    async with _environment(settings=settings) as environment:
        environment.backend.submit_error = OSError("engine unavailable")
        created = await environment.client.responses.create(
            model="background", input="Hello", background=True
        )
        started = await environment.client.responses.create(
            model="background", input="Running", background=True
        )
        await environment.store.mark_in_progress(started.id, now=datetime.now(UTC))
        environment.backend.submit_error = None

        upkeep = await environment.worker.maintain(resubmit=environment.backend.submit)
        job = await environment.backend.receive()
        assert job is not None
        await environment.worker.execute(job)

        completed = await environment.client.responses.retrieve(created.id)
        assert created.status == "queued"
        assert upkeep["resubmitted"] == 1
        assert job == created.id
        assert await environment.backend.receive() is None
        assert completed.status == "completed"


async def test_cancellation_stops_the_run_and_survives_a_stop_failure() -> None:
    async with _environment() as environment:
        stopped = await environment.client.responses.create(
            model="background", input="Hello", background=True
        )
        unstoppable = await environment.client.responses.create(
            model="background", input="Hello", background=True
        )

        await environment.client.responses.cancel(stopped.id)

        async def fail_stop(_run: StoredRun) -> None:
            message = "engine unavailable"
            raise OSError(message)

        environment.backend.stop = fail_stop
        cancelled = await environment.client.responses.cancel(unstoppable.id)

        assert environment.backend.stopped == [stopped.id]
        assert cancelled.status == "cancelled"


async def test_cancellation_wins_a_race_with_graph_publication() -> None:
    started = Event()
    allow_execution = Event()
    async with _environment(execution_gate=(started, allow_execution)) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        async with create_task_group() as tasks:
            tasks.start_soon(environment.worker.execute, job)
            await started.wait()
            cancelled = await environment.client.responses.cancel(created.id)
            allow_execution.set()

        current = await environment.client.responses.retrieve(created.id)
        assert cancelled.status == "cancelled"
        assert current.status == "cancelled"


async def test_in_memory_backend_cancels_active_execution() -> None:
    started = Event()
    allow_execution = Event()
    async with _environment(execution_gate=(started, allow_execution)) as environment:
        async with _in_memory_client(
            environment,
            maintenance_interval=timedelta(milliseconds=10),
        ) as (client, backend):
            created = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )
            await started.wait()
            cancelled = await client.responses.cancel(created.id)
            with fail_after(2):
                while True:
                    stored = await backend.store.get(created.id)
                    if stored is not None and not stored.cleanup_pending:
                        break
                    await checkpoint()

        assert cancelled.status == "cancelled"
        assert stored.status == "cancelled"


async def test_in_memory_backend_preserves_stored_cancellation_retention() -> None:
    started = Event()
    allow_execution = Event()
    settings = BackgroundSettings(
        result_retention=timedelta(seconds=1),
        stored_result_retention=timedelta(days=30),
    )
    async with _environment(execution_gate=(started, allow_execution)) as environment:
        async with _in_memory_client(
            environment,
            settings=settings,
        ) as (client, backend):
            created = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
                store=True,
            )
            await started.wait()
            await client.responses.cancel(created.id)
            stored = await backend.store.get(created.id)

        assert stored is not None
        assert stored.result_expires_at is not None
        assert stored.result_expires_at - datetime.now(UTC) > timedelta(days=29)


async def test_background_rejects_interrupt_run_id_metadata() -> None:
    async with _environment() as environment:
        with pytest.raises(BadRequestError) as error:
            await environment.client.responses.create(
                model="background",
                input="Hello",
                background=True,
                metadata={"lgos_run_id": str(uuid.uuid4())},
            )
        assert error.value.response.json()["error"]["param"] == ("metadata.lgos_run_id")
        assert await environment.backend.receive() is None


async def test_idempotency_key_replays_the_original_response() -> None:
    headers = {"Idempotency-Key": "create-report-1"}
    async with _environment() as environment:
        first = await environment.client.responses.create(
            model="background", input="Hello", background=True, extra_headers=headers
        )
        replay = await environment.client.responses.create(
            model="background", input="Hello", background=True, extra_headers=headers
        )
        other_owner = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            extra_headers={**headers, "x-owner": "tenant-b"},
        )
        with pytest.raises(UnprocessableEntityError) as reused:
            await environment.client.responses.create(
                model="background",
                input="Different",
                background=True,
                extra_headers=headers,
            )

        assert replay == first
        assert other_owner.id != first.id
        assert reused.value.response.json()["error"]["param"] == "Idempotency-Key"
        submitted = {await environment.backend.receive() for _ in range(3)}
        assert submitted == {first.id, other_owner.id, None}


@pytest.mark.parametrize("key", ["", "k" * 256])
async def test_invalid_idempotency_key_is_rejected(key: str) -> None:
    async with _environment() as environment:
        with pytest.raises(BadRequestError) as error:
            await environment.client.responses.create(
                model="background",
                input="Hello",
                background=True,
                extra_headers={"Idempotency-Key": key},
            )
        assert error.value.response.json()["error"]["param"] == "Idempotency-Key"


async def test_in_memory_backend_executes_an_idempotent_replay_once() -> None:
    headers = {"Idempotency-Key": "create-report-1"}
    async with _environment() as environment:
        async with _in_memory_client(environment) as (client, _backend):
            first = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
                extra_headers=headers,
            )
            replay = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
                extra_headers=headers,
            )
            completed = await _terminal_response(client, first.id)

        assert replay.id == first.id
        assert completed.status == "completed"
        assert environment.invocations == ["Hello"]


@pytest.mark.parametrize("render_error", [None, ValueError("broken renderer")])
async def test_lost_lease_preserves_background_state_for_recovery(render_error):
    leases = []

    @asynccontextmanager
    async def coordinator(_key):
        lease = RunLease()
        leases.append(lease)
        yield lease

    started, proceed = Event(), Event()
    async with _environment(
        coordinator=coordinator,
        execution_gate=(started, proceed),
        render_error=render_error,
    ) as environment:
        created = await environment.client.responses.create(
            model="background", input="Hello", background=True
        )

        async def execute():
            with pytest.raises(RunLeaseLostError):
                await environment.worker.execute(created.id)

        with fail_after(5):
            async with create_task_group() as tasks:
                tasks.start_soon(execute)
                await started.wait()
                # Model a late graph result after ownership has been revoked.
                leases[0].lost = True
                proceed.set()

        current = await environment.client.responses.retrieve(created.id)
        assert current.status == "in_progress"
        stored = await environment.store.get(created.id)
        assert stored is not None
        graph = await environment.worker.graphs.get_graph("background").resolve_graph()
        checkpoint_config = {"configurable": {"thread_id": stored.checkpoint_thread_id}}
        assert await graph.checkpointer.aget_tuple(checkpoint_config) is not None

        await environment.worker.finalize(created.id)
        final = await environment.client.responses.retrieve(created.id)
        assert final.status == ("failed" if render_error else "completed")
        assert environment.invocations == ["Hello"]
        assert await graph.checkpointer.aget_tuple(checkpoint_config) is None


async def test_lease_loss_after_publication_defers_cleanup_to_a_new_owner():
    leases = []

    @asynccontextmanager
    async def coordinator(_key):
        lease = RunLease()
        leases.append(lease)
        yield lease

    class LosingStore(InMemoryResponseStore):
        async def finish(self, *args, **kwargs):
            result = await super().finish(*args, **kwargs)
            leases[-1].lost = True
            return result

    async with _environment(
        coordinator=coordinator, response_store=LosingStore()
    ) as environment:
        created = await environment.client.responses.create(
            model="background", input="Hello", background=True
        )
        with pytest.raises(RunLeaseLostError):
            await environment.worker.execute(created.id)

        stored = await environment.store.get(created.id)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.cleanup_pending
        graph = await environment.worker.graphs.get_graph("background").resolve_graph()
        checkpoint_config = {"configurable": {"thread_id": stored.checkpoint_thread_id}}
        assert await graph.checkpointer.aget_tuple(checkpoint_config) is not None

        assert (await environment.worker.maintain())["cleaned"] == 1
        assert await graph.checkpointer.aget_tuple(checkpoint_config) is None
        cleaned = await environment.store.get(created.id)
        assert cleaned is not None
        assert not cleaned.cleanup_pending


async def test_worker_retry_resumes_from_the_checkpoint() -> None:
    async with _environment(fail_after_checkpoint=1) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        with pytest.raises(OSError, match="injected graph failure"):
            await environment.worker.execute(job)
        await environment.worker.execute(job)

        completed = await environment.client.responses.retrieve(created.id)
        assert completed.status == "completed"
        assert environment.invocations == ["Hello"]


async def test_worker_finalization_publishes_failed_response() -> None:
    async with _environment(fail_after_checkpoint=2) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        with pytest.raises(OSError, match="injected graph failure"):
            await environment.worker.execute(job)
        await environment.worker.finalize(job)

        failed = await environment.client.responses.retrieve(created.id)
        assert failed.status == "failed"
        assert failed.error is not None
        assert "ended before" in failed.error.message


async def test_output_renderer_failure_is_terminal() -> None:
    async with _environment(render_error=ValueError("broken renderer")) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        await environment.worker.execute(job)

        failed = await environment.client.responses.retrieve(created.id)
        assert failed.status == "failed"
        assert failed.error is not None
        assert "configuration" in failed.error.message


async def test_invalid_persisted_request_still_publishes_failure() -> None:
    settings = BackgroundSettings(
        result_retention=timedelta(seconds=1),
        stored_result_retention=timedelta(days=30),
    )
    async with _environment(settings=settings) as environment:
        await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            store=True,
        )
        job = await environment.backend.receive()
        assert job is not None
        original = await environment.store.get(job)
        assert original is not None
        graph = await environment.worker.graphs.get_graph("background").resolve_graph()
        assert isinstance(graph.checkpointer, AsyncSqliteSaver)
        await graph.checkpointer.setup()

        payload = {field: getattr(original, field) for field in NewRun.model_fields}
        payload["envelope"] = {}
        store = InMemoryResponseStore()
        await store.create(NewRun.model_validate(payload))
        worker = BackgroundWorker(
            graphs=environment.worker.graphs,
            store=store,
            settings=settings,
        )

        await worker.execute(job)

        failed = await store.get(job)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.response["status"] == "failed"
        assert failed.result_expires_at is not None
        assert failed.result_expires_at - datetime.now(UTC) > timedelta(days=29)


async def test_removed_graph_abandons_unresolvable_cleanup() -> None:
    async with _environment() as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None
        replacement = environment.worker.graphs.get_graph("background")
        worker = BackgroundWorker(
            graphs=GraphRegistry(registry={"replacement": replacement}),
            store=environment.store,
        )

        await worker.execute(job)

        failed = await environment.client.responses.retrieve(created.id)
        stored = await environment.store.get(job)
        assert failed.status == "failed"
        assert stored is not None
        assert stored.cleanup_pending is False

        future = datetime.now(UTC) + timedelta(days=31)
        assert await environment.store.expire(now=future, limit=1) == 1
        assert await environment.store.get(job) is None


async def test_invalid_graph_abandons_unresolvable_cleanup() -> None:
    async with _environment() as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None
        graph_config = environment.worker.graphs.get_graph("background")

        def invalid_graph():
            message = "invalid graph"
            raise GraphConfigurationError(message)

        worker = BackgroundWorker(
            graphs=GraphRegistry(
                registry={
                    "background": graph_config.model_copy(
                        update={"graph": invalid_graph}
                    )
                }
            ),
            store=environment.store,
        )

        await worker.execute(job)

        failed = await environment.client.responses.retrieve(created.id)
        stored = await environment.store.get(job)
        assert failed.status == "failed"
        assert stored is not None
        assert stored.cleanup_pending is False


async def test_unknown_and_cursor_retrieval_use_openai_errors() -> None:
    async with _environment() as environment:
        missing = await environment.http.get("/v1/responses/resp_unknown")
        cursor = await environment.http.get(
            "/v1/responses/resp_unknown",
            params={"starting_after": "item"},
        )

        assert missing.status_code == status.HTTP_404_NOT_FOUND
        assert missing.json()["error"]["code"] == "response_not_found"
        assert cursor.status_code == status.HTTP_400_BAD_REQUEST
        assert cursor.json()["error"]["param"] == "starting_after"
