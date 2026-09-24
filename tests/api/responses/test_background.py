from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from anyio import Event, fail_after, get_cancelled_exc_class
from anyio.lowlevel import checkpoint
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from openai import (
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)

from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    InMemoryBackgroundBackend,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from tests.api.interrupt.support import resume_outputs
from tests.graph.support.interrupt import make_multi_interrupt_graph
from tests.graph.support.schemas import MessageState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import Request
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from openai.types.responses import Response

MODEL = "background"


@dataclass
class _Graph:
    """A one-node graph whose runs tests can hold, fail, and count."""

    started: Event = field(default_factory=Event)
    stopped: Event = field(default_factory=Event)
    release: Event | None = None
    error: Exception | None = None
    inputs: list[str] = field(default_factory=list)

    async def reply(self, state: MessageState):
        self.inputs.append(state["messages"][-1].text)
        self.started.set()
        if self.release is not None:
            try:
                await self.release.wait()
            except get_cancelled_exc_class():
                self.stopped.set()
                raise
        if self.error is not None:
            raise self.error
        return {"messages": [AIMessage(content=f"done:{self.inputs[-1]}")]}

    def config(self, **options) -> GraphConfig:
        graph = (
            StateGraph(MessageState)
            .add_node("reply", self.reply)
            .set_entry_point("reply")
            .set_finish_point("reply")
            .compile()
        )
        return GraphConfig(
            graph=graph,
            description="Background test graph",
            **{"features": {GraphFeature.BACKGROUND}, **options},
        )


@asynccontextmanager
async def _client(
    registry: GraphRegistry,
    *,
    configured: bool = True,
) -> AsyncIterator[AsyncOpenAI]:
    backend = InMemoryBackgroundBackend(registry)
    app = FastAPI(lifespan=backend.lifespan)

    def checkpoint_scope(request: Request) -> str:
        return request.headers.get("x-owner", "tenant-a")

    LanggraphOpenaiServe(
        app=app,
        graphs=registry,
        checkpoint_scope=checkpoint_scope,
        background=backend if configured else None,
    ).bind_openai_api()
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http,
            max_retries=0,
        ) as client,
    ):
        yield client


async def _finished(
    client: AsyncOpenAI,
    response_id: str,
    headers: dict[str, str] | None = None,
) -> Response:
    with fail_after(2):
        while True:
            response = await client.responses.retrieve(
                response_id, extra_headers=headers
            )
            if response.status not in {"queued", "in_progress"}:
                return response
            await checkpoint()


async def test_background_response_runs_after_create_returns() -> None:
    graph = _Graph(release=Event())
    async with _client(GraphRegistry(registry={MODEL: graph.config()})) as client:
        created = await client.responses.create(
            model=MODEL, input="Hello", background=True
        )
        with fail_after(2):
            await graph.started.wait()
        running = await client.responses.retrieve(created.id)
        graph.release.set()
        completed = await _finished(client, created.id)

    assert created.status == "queued"
    assert created.background is True
    assert running.status == "in_progress"
    assert completed.status == "completed"
    assert completed.output_text == "done:Hello"
    assert (completed.id, completed.created_at) == (created.id, created.created_at)


async def test_background_create_requires_a_backend_and_an_opted_in_model() -> None:
    registry = GraphRegistry(
        registry={
            MODEL: _Graph().config(),
            "foreground-only": _Graph().config(features=set()),
        }
    )
    async with _client(registry, configured=False) as client:
        with pytest.raises(BadRequestError) as unconfigured:
            await client.responses.create(model=MODEL, input="Hi", background=True)
    async with _client(registry) as client:
        with pytest.raises(BadRequestError) as not_opted_in:
            await client.responses.create(
                model="foreground-only", input="Hi", background=True
            )
        foreground = await client.responses.create(model=MODEL, input="Hi")

    assert unconfigured.value.param == not_opted_in.value.param == "background"
    assert foreground.output_text == "done:Hi"


@pytest.mark.parametrize(
    ("options", "param"),
    [
        pytest.param({"stream": True}, "stream", id="stream"),
        pytest.param(
            {"metadata": {"lgos_settings": '{"count":"invalid"}'}},
            "metadata.lgos_settings",
            id="client-settings",
        ),
        pytest.param(
            {"metadata": {"lgos_run_id": "7d2c6f3e-4b1a-4e8f-9c0d-2a5b3e6f7a81"}},
            "metadata.lgos_run_id",
            id="run-id",
        ),
    ],
)
async def test_invalid_background_requests_are_rejected_before_execution(
    options, param
) -> None:
    class Settings(ClientSettings):
        count: int = 1

    graph = _Graph()
    registry = GraphRegistry(registry={MODEL: graph.config(client_settings=Settings)})
    async with _client(registry) as client:
        with pytest.raises(BadRequestError) as error:
            await client.responses.create(
                model=MODEL, input="Hello", background=True, **options
            )

    assert error.value.param == param
    assert graph.inputs == []


async def test_retrieval_is_owner_scoped_and_rejects_streaming() -> None:
    async with _client(GraphRegistry(registry={MODEL: _Graph().config()})) as client:
        created = await client.responses.create(
            model=MODEL, input="Hello", background=True
        )
        await _finished(client, created.id)
        other_owner = {"x-owner": "tenant-b"}
        with pytest.raises(NotFoundError):
            await client.responses.retrieve(created.id, extra_headers=other_owner)
        with pytest.raises(NotFoundError):
            await client.responses.cancel(created.id, extra_headers=other_owner)
        with pytest.raises(NotFoundError) as unknown:
            await client.responses.retrieve(
                f"{created.id.rsplit('_', 1)[0]}_{uuid.uuid4().hex}"
            )
        with pytest.raises(BadRequestError) as streaming:
            await client.responses.retrieve(created.id, stream=True)

    assert unknown.value.code == "response_not_found"
    assert streaming.value.param == "stream"


async def test_cancellation_stops_an_active_run_and_keeps_a_finished_one() -> None:
    graph = _Graph(release=Event())
    async with _client(GraphRegistry(registry={MODEL: graph.config()})) as client:
        active = await client.responses.create(
            model=MODEL, input="Hello", background=True
        )
        with fail_after(2):
            await graph.started.wait()
        cancelled = await client.responses.cancel(active.id)
        with fail_after(2):
            await graph.stopped.wait()
        after_cancel = await client.responses.retrieve(active.id)
        graph.release.set()
        finished = await client.responses.create(
            model=MODEL, input="Again", background=True
        )
        completed = await _finished(client, finished.id)
        cancelled_again = await client.responses.cancel(finished.id)

    assert cancelled.status == after_cancel.status == "cancelled"
    assert cancelled_again == completed


async def test_unexpected_graph_failure_fails_the_response() -> None:
    graph = _Graph(error=OSError("disk full"))
    async with _client(GraphRegistry(registry={MODEL: graph.config()})) as client:
        created = await client.responses.create(
            model=MODEL, input="Hello", background=True
        )
        failed = await _finished(client, created.id)

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "server_error"
    # Internal exception text never reaches the client.
    assert "disk full" not in failed.error.message


async def test_idempotency_key_returns_the_first_response() -> None:
    headers = {"Idempotency-Key": "create-report-1"}
    graph = _Graph()
    async with _client(GraphRegistry(registry={MODEL: graph.config()})) as client:
        first = await client.responses.create(
            model=MODEL, input="Hello", background=True, extra_headers=headers
        )
        await _finished(client, first.id)
        replay = await client.responses.create(
            model=MODEL, input="Hello", background=True, extra_headers=headers
        )
        other_headers = {**headers, "x-owner": "tenant-b"}
        other_owner = await client.responses.create(
            model=MODEL, input="Hello", background=True, extra_headers=other_headers
        )
        with pytest.raises(UnprocessableEntityError) as reused:
            await client.responses.create(
                model=MODEL, input="Different", background=True, extra_headers=headers
            )
        with pytest.raises(BadRequestError) as too_long:
            await client.responses.create(
                model=MODEL,
                input="Hello",
                background=True,
                extra_headers={"Idempotency-Key": "k" * 256},
            )
        await _finished(client, other_owner.id, other_headers)

    assert replay.id == first.id
    assert replay.status == "completed"
    assert other_owner.id != first.id
    assert reused.value.code == "idempotency_key_reused"
    assert too_long.value.param == "Idempotency-Key"
    assert graph.inputs == ["Hello", "Hello"]


async def test_one_of_two_answers_to_a_pause_continues_the_run(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = GraphRegistry(
        registry={
            "approval": GraphConfig(
                graph=make_multi_interrupt_graph(sqlite_checkpointer),
                description="Two questions",
                features={GraphFeature.INTERRUPTS, GraphFeature.BACKGROUND},
                run_coordinator=InMemoryRunCoordinator(),
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_message=lambda output: AIMessage(
                    content=",".join(output["answers"])
                ),
            )
        }
    )
    async with _client(registry) as client:
        created = await client.responses.create(
            model="approval", input="Hi", background=True
        )
        paused = await _finished(client, created.id)

        # Two reviewers answer the same pause before either answer finishes.
        created_answers = {
            value: await client.responses.create(
                model="approval",
                previous_response_id=paused.id,
                input=resume_outputs(paused, [value]),
                background=True,
            )
            for value in ("approve", "reject")
        }
        answers = {
            value: await _finished(client, answer.id)
            for value, answer in created_answers.items()
        }
        (winner,) = [key for key, value in answers.items() if value.error is None]
        # The run continues in either mode from the winning answer's pause.
        final = await client.responses.create(
            model="approval",
            previous_response_id=answers[winner].id,
            input=resume_outputs(answers[winner], ["second"]),
        )

    assert paused.status == "completed"
    assert [item.name for item in paused.output] == ["lgos_interrupt"]
    assert {response.status for response in answers.values()} == {
        "completed",
        "failed",
    }
    assert final.output_text == f"{winner},second"
