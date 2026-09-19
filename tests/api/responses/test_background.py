from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from anyio import Event, create_task_group
from fastapi import status
from httpx2 import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from openai import AsyncOpenAI, BadRequestError, ConflictError, RateLimitError

from langgraph_openai_serve import (
    BackgroundPolicy,
    BackgroundSettings,
    BackgroundWorker,
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
    RetryableJobError,
    RunJob,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from tests.background.fakes import MemoryBackgroundBackend, MemoryResponseStore
from tests.graph.support.schemas import MessageState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import Request


@dataclass
class _Environment:
    client: AsyncOpenAI
    http: AsyncClient
    backend: MemoryBackgroundBackend
    worker: BackgroundWorker
    store: MemoryResponseStore
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
                )
            }
        )
        store = MemoryResponseStore()
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


async def test_background_response_is_polled_and_executed_outside_post() -> None:
    async with _environment() as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )

        assert created.status == "queued"
        assert created.background is True
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


async def test_idempotent_create_reuses_response_and_rejects_conflicts() -> None:
    key = str(uuid.uuid4())
    async with _environment() as environment:
        first = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            metadata={"lgos_run_id": key},
        )
        replay = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
            metadata={"lgos_run_id": key},
        )

        assert replay == first
        with pytest.raises(ConflictError):
            await environment.client.responses.create(
                model="background",
                input="Different",
                background=True,
                metadata={"lgos_run_id": key},
            )
        assert await environment.backend.receive() is not None
        assert await environment.backend.receive() is None


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


async def test_incompatible_hatchet_envelope_fails_the_public_response() -> None:
    async with _environment() as environment:
        created = await environment.client.responses.create(
            model="background",
            input="Hello",
            background=True,
        )
        job = await environment.backend.receive()
        assert job is not None

        await environment.worker.execute(RunJob(run_id=job.run_id, schema_version=2))

        failed = await environment.client.responses.retrieve(created.id)
        assert failed.status == "failed"
        assert failed.error is not None
        assert "envelope version" in failed.error.message


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
