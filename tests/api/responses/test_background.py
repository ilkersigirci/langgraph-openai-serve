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
from openai import AsyncOpenAI, BadRequestError, ConflictError, RateLimitError

from langgraph_openai_serve import (
    BackgroundPolicy,
    BackgroundSettings,
    BackgroundWorker,
    ClientSettings,
    GraphConfig,
    GraphRegistry,
    InMemoryBackgroundBackend,
    InMemoryResponseStore,
    LanggraphOpenaiServe,
    NewRun,
    RetryableJobError,
    StoredRun,
)
from langgraph_openai_serve.graph.graph_registry import GraphConfigurationError
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
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
                    background=(
                        BackgroundPolicy(version="test-v1") if model_enabled else None
                    ),
                    run_coordinator=(
                        InMemoryRunCoordinator() if model_enabled else None
                    ),
                    output_to_message=render_output if render_error else None,
                )
            }
        )
        store = InMemoryResponseStore()
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
) -> AsyncIterator[tuple[AsyncOpenAI, InMemoryBackgroundBackend]]:
    backend = InMemoryBackgroundBackend(
        graphs=environment.worker.graphs,
        settings=settings,
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


async def test_in_memory_backend_finishes_a_cancelled_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _environment() as environment:
        async with _in_memory_client(environment) as (client, backend):
            receipt_started = Event()
            release_receipt = Event()
            response_ids: list[str] = []
            record_workflow_run = backend.store.record_workflow_run

            async def delayed_receipt(
                response_id: str,
                workflow_run_id: str,
                *,
                now: datetime,
            ) -> StoredRun | None:
                response_ids.append(response_id)
                receipt_started.set()
                await release_receipt.wait()
                return await record_workflow_run(
                    response_id,
                    workflow_run_id,
                    now=now,
                )

            monkeypatch.setattr(
                backend.store,
                "record_workflow_run",
                delayed_receipt,
            )
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
                    await receipt_started.wait()
                    request_scope.cancel()
                    await checkpoint()
                    release_receipt.set()

            assert len(response_ids) == 1
            stored = await backend.store.get_internal(response_ids[0])
            assert stored is not None
            assert stored.workflow_run_id == stored.response_id
            completed = await _terminal_response(client, stored.response_id)

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
async def test_invalid_background_settings_are_rejected_before_admission(encoded):
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
        async with _in_memory_client(environment) as (client, backend):
            created = await client.responses.create(
                model="background",
                input="Hello",
                background=True,
            )
            await started.wait()
            cancelled = await client.responses.cancel(created.id)
            with fail_after(2):
                while True:
                    stored = await backend.store.get_internal(created.id)
                    if (
                        stored is not None
                        and not stored.cancellation_pending
                        and not stored.cleanup_pending
                    ):
                        break
                    await checkpoint()

        assert cancelled.status == "cancelled"
        assert stored is not None
        assert not stored.cancellation_pending
        assert not stored.cleanup_pending


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
            stored = await backend.store.get_internal(created.id)

        assert stored is not None
        assert stored.result_expires_at is not None
        assert stored.terminal_at is not None
        assert (
            stored.result_expires_at - stored.terminal_at
            == settings.stored_result_retention
        )


async def test_cancellation_uses_the_persisted_response_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = BackgroundSettings(
        result_retention=timedelta(seconds=1),
        stored_result_retention=timedelta(days=30),
    )
    async with _environment(settings=settings) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            store=True,
        )
        retrieve = environment.backend.retrieve

        async def retrieve_without_envelope(
            response_id: str,
            owner_scope: str,
        ) -> StoredRun | None:
            current = await retrieve(response_id, owner_scope)
            return (
                current.model_copy(update={"envelope": {}})
                if current is not None
                else None
            )

        monkeypatch.setattr(
            environment.backend,
            "retrieve",
            retrieve_without_envelope,
        )

        cancelled = await environment.client.responses.cancel(created.id)
        stored = await environment.store.get_internal(created.id)

        assert cancelled.status == "cancelled"
        assert stored is not None
        assert stored.result_expires_at is not None
        assert stored.terminal_at is not None
        assert (
            stored.result_expires_at - stored.terminal_at
            == settings.stored_result_retention
        )


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


async def test_idempotency_header_reuses_response_and_rejects_conflicts() -> None:
    headers = {"Idempotency-Key": "create-background-report"}
    async with _environment() as environment:
        first = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            extra_headers=headers,
        )
        replay = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            extra_headers=headers,
        )
        stored = await environment.store.get_internal(first.id)

        assert replay == first
        assert stored is not None
        assert stored.idempotency_digest == (
            "252f3fe49ded85e663315f02074be81b747f7be603889b989967858a36072ed0"
        )
        assert headers["Idempotency-Key"] not in stored.model_dump_json()
        with pytest.raises(ConflictError) as error:
            await environment.client.responses.create(
                model="background",
                input="Different",
                background=True,
                extra_headers=headers,
            )
        assert error.value.response.json()["error"]["param"] == "Idempotency-Key"
        assert await environment.backend.receive() is not None
        assert await environment.backend.receive() is None


async def test_idempotency_header_is_owner_scoped() -> None:
    headers = {"Idempotency-Key": "shared-client-key"}
    async with _environment() as environment:
        first = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            extra_headers=headers,
        )
        other_owner = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            extra_headers={**headers, "x-owner": "tenant-b"},
        )

        assert other_owner.id != first.id


async def test_admission_capacity_counts_only_active_responses() -> None:
    async with _environment(
        settings=BackgroundSettings(admission_capacity=1)
    ) as environment:
        first = await environment.client.responses.create(
            model="background",
            input="First",
            background=True,
        )
        with pytest.raises(RateLimitError):
            await environment.client.responses.create(
                model="background",
                input="Second",
                background=True,
            )

        await _execute_next(environment)
        second = await environment.client.responses.create(
            model="background",
            input="Second",
            background=True,
        )
        assert second.id != first.id


async def test_hatchet_retry_resumes_from_the_checkpoint() -> None:
    async with _environment(fail_after_checkpoint=1) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        with pytest.raises(RetryableJobError):
            await environment.worker.execute(job)
        await environment.worker.execute(job)

        completed = await environment.client.responses.retrieve(created.id)
        assert completed.status == "completed"
        assert environment.invocations == ["Hello"]


async def test_hatchet_failure_hook_publishes_failed_response() -> None:
    async with _environment(fail_after_checkpoint=2) as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        with pytest.raises(RetryableJobError):
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
        original = await environment.store.get_internal(job)
        assert original is not None
        graph = await environment.worker.graphs.get_graph("background").resolve_graph()
        assert isinstance(graph.checkpointer, AsyncSqliteSaver)
        await graph.checkpointer.setup()

        payload = {field: getattr(original, field) for field in NewRun.model_fields}
        payload["envelope"] = {}
        store = InMemoryResponseStore()
        await store.accept(NewRun.model_validate(payload), capacity=1)
        worker = BackgroundWorker(
            graphs=environment.worker.graphs,
            store=store,
            settings=settings,
        )

        await worker.execute(job)

        failed = await store.get_internal(job)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.response is not None
        assert failed.response["status"] == "failed"
        assert failed.result_expires_at is not None
        assert failed.terminal_at is not None
        assert (
            failed.result_expires_at - failed.terminal_at
            == settings.stored_result_retention
        )


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
        stored = await environment.store.get_internal(job)
        assert failed.status == "failed"
        assert stored is not None
        assert stored.cleanup_pending is False

        future = datetime.now(UTC) + timedelta(days=31)
        assert await environment.store.expire(now=future, limit=1) == 1
        assert await environment.store.get_internal(job) is None


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
        stored = await environment.store.get_internal(job)
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
