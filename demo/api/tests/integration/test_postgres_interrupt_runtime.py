"""Exercise the complete interrupt API contract against real PostgreSQL."""

import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from uuid import uuid4

import anyio
import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe
from langgraph_openai_serve.api.responses.interrupts import interrupt_run_id
from langgraph_openai_serve.graph.interrupt import checkpoint_key
from langgraph_openai_serve.server import ServerResources, ServerSettings
from langgraph_openai_serve.server.runtime import open_resources
from openai import AsyncOpenAI, ConflictError
from openai.types.responses import ResponseFunctionToolCall
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import DivisionByZero
from pydantic import SecretStr

from lgos_demo_api.graphs.interruptible import (
    create_interruptible_graph,
    create_interruptible_graph_config,
)

POSTGRES_URI = os.environ.get("DEMO_API_TEST_POSTGRES_URI")
MODEL = "interruptible-approval"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        POSTGRES_URI is None,
        reason="DEMO_API_TEST_POSTGRES_URI is required",
    ),
]


def _resources(postgres_uri: str) -> AbstractAsyncContextManager[ServerResources]:
    """Open the PostgreSQL resources that `lgos serve` gives the registry."""
    return open_resources(ServerSettings(POSTGRES_URI=SecretStr(postgres_uri)))


def _api_app(runtime: ServerResources) -> FastAPI:
    graph = create_interruptible_graph(runtime.checkpointer)
    registry = GraphRegistry(
        graphs={MODEL: create_interruptible_graph_config(lambda: graph)},
        run_coordinator=runtime.run_coordinator,
    )
    return LanggraphOpenaiServe(registry=registry).bind_openai_api().app


@asynccontextmanager
async def _openai_client(runtime: ServerResources) -> AsyncGenerator[AsyncOpenAI, None]:
    async with (
        AsyncClient(
            transport=ASGITransport(app=_api_app(runtime)),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as client,
    ):
        yield client


@pytest.fixture
async def postgres_threads() -> AsyncIterator[list[str]]:
    """Delete the checkpoint threads a test records, whatever its outcome."""
    assert POSTGRES_URI is not None
    threads: list[str] = []
    try:
        yield threads
    finally:
        async with _resources(POSTGRES_URI) as runtime:
            for thread_id in threads:
                await runtime.checkpointer.adelete_thread(thread_id)


async def test_openai_interrupt_survives_restart_and_excludes_another_worker(
    postgres_threads: list[str],
) -> None:
    """Resume after restart, reject overlap, and delete terminal state."""
    assert POSTGRES_URI is not None
    public_request = "Refund order ORDER-PG"

    async with (
        _resources(POSTGRES_URI) as initial_runtime,
        _openai_client(initial_runtime) as client,
    ):
        paused = await client.responses.create(
            store=False,
            model=MODEL,
            input=[{"role": "user", "content": public_request}],
        )
    checkpoint_thread_id = checkpoint_key(MODEL, interrupt_run_id(paused.id))
    postgres_threads.append(checkpoint_thread_id)

    tool_calls = [
        item for item in paused.output if isinstance(item, ResponseFunctionToolCall)
    ]
    assert len(tool_calls) == 1
    arguments = json.loads(tool_calls[0].arguments)
    assert tool_calls[0].call_id.startswith("call_lg_")
    assert arguments["action"] == "refund"
    resume_items = [
        {
            "type": "function_call_output",
            "call_id": tool_calls[0].call_id,
            "output": "approve",
        },
    ]

    async with (
        _resources(POSTGRES_URI) as lock_runtime,
        _resources(POSTGRES_URI) as first_resume_runtime,
        _openai_client(first_resume_runtime) as client,
    ):
        async with lock_runtime.run_coordinator(checkpoint_thread_id):
            with pytest.raises(ConflictError) as exc_info:
                await client.responses.create(
                    store=False,
                    model=MODEL,
                    previous_response_id=paused.id,
                    input=resume_items,
                )
            assert exc_info.value.code == "run_busy"

        completed = await client.responses.create(
            store=False,
            model=MODEL,
            previous_response_id=paused.id,
            input=resume_items,
        )

        assert completed.output_text == (
            f"Review workflow for: {public_request}\n"
            "- Refund: approve\n"
            "- Customer notification: sent\n"
            "- Executed actions: Refund, Customer notification"
        )
        config = {"configurable": {"thread_id": checkpoint_thread_id}}
        assert await first_resume_runtime.checkpointer.aget_tuple(config) is None


@pytest.fixture
async def empty_postgres_schema() -> AsyncIterator[str]:
    """Own a fresh schema without touching another test's persistence data."""
    assert POSTGRES_URI is not None
    schema = f"startup_{uuid4().hex}"
    async with await AsyncConnection.connect(
        POSTGRES_URI, autocommit=True
    ) as connection:
        await connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
        try:
            yield make_conninfo(POSTGRES_URI, options=f"-csearch_path={schema}")
        finally:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


async def test_concurrent_startup_migrates_empty_schema_and_restart_preserves_data(
    empty_postgres_schema: str,
) -> None:
    """Concurrent startup initializes persistence, and restart preserves its data."""
    namespace = ("startup",)
    start = anyio.Event()

    async def start_replica(key: str) -> None:
        await start.wait()
        async with _resources(empty_postgres_schema) as runtime:
            await runtime.store.aput(namespace, key, {"started": True})
            assert (
                await runtime.checkpointer.aget_tuple(
                    {"configurable": {"thread_id": key}}
                )
                is None
            )

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tasks:
            for key in ("api-a", "api-b", "worker"):
                tasks.start_soon(start_replica, key)
            start.set()

        async with _resources(empty_postgres_schema) as runtime:
            items = await runtime.store.asearch(namespace)
            assert {item.key: item.value for item in items} == {
                "api-a": {"started": True},
                "api-b": {"started": True},
                "worker": {"started": True},
            }


async def test_failed_migration_releases_lock_for_the_next_startup(
    empty_postgres_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            AsyncPostgresStore,
            "MIGRATIONS",
            [*AsyncPostgresStore.MIGRATIONS, "SELECT 1 / 0;"],
        )
        with pytest.raises(DivisionByZero):
            async with _resources(empty_postgres_schema):
                pytest.fail("A failed migration must prevent startup")

    with anyio.fail_after(10):
        async with _resources(empty_postgres_schema) as runtime:
            await runtime.store.aput(("startup",), "retried", {"ready": True})
            item = await runtime.store.aget(("startup",), "retried")
            assert item is not None
            assert item.value == {"ready": True}
