"""Exercise the complete interrupt API contract against real PostgreSQL."""

import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe
from langgraph_openai_serve.graph.utils import checkpoint_key
from openai import AsyncOpenAI, ConflictError

from lgos_demo_api.checkpointer import (
    PostgresRuntime,
    postgres_runtime,
    setup_postgres_schema,
)
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


def _api_app(runtime: PostgresRuntime) -> FastAPI:
    graph = create_interruptible_graph(runtime.checkpointer)
    registry = GraphRegistry(
        registry={
            MODEL: create_interruptible_graph_config(
                lambda: graph,
                runtime.run_coordinator,
            )
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


@asynccontextmanager
async def _openai_client(runtime: PostgresRuntime) -> AsyncIterator[AsyncOpenAI]:
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
async def postgres_run_identity() -> AsyncIterator[tuple[str, str]]:
    assert POSTGRES_URI is not None
    await setup_postgres_schema(POSTGRES_URI)
    run_id = str(uuid.uuid4())
    checkpoint_thread_id = checkpoint_key(MODEL, run_id)

    try:
        yield run_id, checkpoint_thread_id
    finally:
        async with postgres_runtime(POSTGRES_URI) as runtime:
            await runtime.checkpointer.adelete_thread(checkpoint_thread_id)


async def test_openai_interrupt_survives_restart_and_excludes_another_worker(
    postgres_run_identity: tuple[str, str],
) -> None:
    """Pause through /v1, restart, reject a competing run, resume, and delete."""
    assert POSTGRES_URI is not None
    run_id, checkpoint_thread_id = postgres_run_identity
    public_request = "Refund order ORDER-PG"

    async with (
        postgres_runtime(POSTGRES_URI) as initial_runtime,
        _openai_client(initial_runtime) as client,
    ):
        paused = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": public_request}],
            metadata={"langgraph_run_id": run_id},
        )

    assistant = paused.choices[0].message
    tool_calls = assistant.tool_calls or []
    assert len(tool_calls) == 2
    arguments = [json.loads(tool_call.function.arguments) for tool_call in tool_calls]
    assert {item["run_id"] for item in arguments} == {run_id}
    assert len({item["state_token"] for item in arguments}) == 1
    resume_messages = [
        assistant.model_dump(mode="json", exclude_none=True),
        *[
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"resume": "approve"}),
            }
            for tool_call in tool_calls
        ],
    ]

    async with (
        postgres_runtime(POSTGRES_URI) as lock_runtime,
        postgres_runtime(POSTGRES_URI) as restarted_runtime,
        _openai_client(restarted_runtime) as client,
    ):
        async with lock_runtime.run_coordinator(checkpoint_thread_id):
            with pytest.raises(ConflictError) as exc_info:
                await client.chat.completions.create(
                    model=MODEL,
                    messages=resume_messages,
                )
            assert exc_info.value.code == "run_busy"

        completed = await client.chat.completions.create(
            model=MODEL,
            messages=resume_messages,
        )

        assert completed.choices[0].message.content == (
            f"Approval results for: {public_request}\n"
            "- Refund: approve\n"
            "- Customer notification: approve"
        )
        config = {"configurable": {"thread_id": checkpoint_thread_id}}
        assert await restarted_runtime.checkpointer.aget_tuple(config) is None
