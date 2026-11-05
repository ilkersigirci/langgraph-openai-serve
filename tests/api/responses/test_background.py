from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from anyio import Event, fail_after, get_cancelled_exc_class
from anyio.lowlevel import checkpoint
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from openai import (
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)
from openai.types.responses import Response

from langgraph_openai_serve import (
    BackgroundJob,
    BackgroundRun,
    ClientSettings,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
    InMemoryBackgroundBackend,
    LanggraphOpenaiServe,
    execute_background_job,
    status_event,
)
from langgraph_openai_serve.api.responses.background import follow_background_response
from langgraph_openai_serve.graph.interrupt import InMemoryRunCoordinator
from tests.api.interrupt.support import resume_outputs
from tests.graph.support.interrupt import (
    make_interrupt_graph,
    make_multi_interrupt_graph,
)
from tests.graph.support.schemas import MessageState

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import Request
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

MODEL = "background"
LOST_EVENT = 5


@dataclass
class _Graph:
    """A one-node graph whose runs tests can hold, fail, and count."""

    started: Event = field(default_factory=Event)
    stopped: Event = field(default_factory=Event)
    release: Event | None = None
    error: Exception | None = None
    finish_reason: str | None = None
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
        answer = AIMessage(
            content=f"done:{self.inputs[-1]}",
            response_metadata={"finish_reason": self.finish_reason},
        )
        return {"messages": [answer]}

    def config(self, **options) -> GraphConfig:
        graph = (
            StateGraph(MessageState, context_schema=options.get("client_settings"))
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
    backend: InMemoryBackgroundBackend | None = None,
) -> AsyncGenerator[AsyncOpenAI, None]:
    backend = backend or InMemoryBackgroundBackend(registry)
    app = FastAPI(lifespan=backend.lifespan)

    def checkpoint_scope(request: Request) -> str:
        return request.headers.get("x-owner", "tenant-a")

    LanggraphOpenaiServe(
        app=app,
        registry=registry,
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
    async with _client(GraphRegistry(graphs={MODEL: graph.config()})) as client:
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
        graphs={
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


async def test_invalid_background_requests_are_rejected_before_execution() -> None:
    class Settings(ClientSettings):
        count: int = 1

    graph = _Graph()
    registry = GraphRegistry(graphs={MODEL: graph.config(client_settings=Settings)})
    async with _client(registry) as client:
        with pytest.raises(BadRequestError) as error:
            await client.responses.create(
                model=MODEL,
                input="Hello",
                background=True,
                metadata={"lgos_settings": '{"count":"invalid"}'},
            )

    assert error.value.param == "metadata.lgos_settings"
    assert graph.inputs == []


async def test_invalid_background_resume_is_rejected_before_execution(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    class Settings(ClientSettings):
        count: int = 1

    registry = GraphRegistry(
        graphs={
            "review": GraphConfig(
                graph=make_interrupt_graph(
                    checkpointer=sqlite_checkpointer, context_schema=Settings
                ),
                description="One question",
                features={GraphFeature.INTERRUPTS, GraphFeature.BACKGROUND},
                client_settings=Settings,
            )
        },
        run_coordinator=InMemoryRunCoordinator(),
    )
    async with _client(registry) as client:
        paused = await client.responses.create(model="review", input="Hi")
        with pytest.raises(BadRequestError) as error:
            await client.responses.create(
                model="review",
                previous_response_id=paused.id,
                input=resume_outputs(paused, ["approve"]),
                background=True,
                metadata={"lgos_settings": '{"count":"invalid"}'},
            )

    assert error.value.param == "metadata.lgos_settings"


async def test_retrieval_is_owner_scoped_and_streams_only_streamed_responses() -> None:
    async with _client(GraphRegistry(graphs={MODEL: _Graph().config()})) as client:
        created = await client.responses.create(
            model=MODEL, input="Hello", background=True
        )
        await _finished(client, created.id)
        other_owner = {"x-owner": "tenant-b"}
        with pytest.raises(NotFoundError):
            await client.responses.retrieve(created.id, extra_headers=other_owner)
        with pytest.raises(NotFoundError):
            await client.responses.retrieve(
                created.id, stream=True, extra_headers=other_owner
            )
        with pytest.raises(NotFoundError):
            await client.responses.cancel(created.id, extra_headers=other_owner)
        with pytest.raises(NotFoundError) as unknown:
            await client.responses.retrieve(
                f"{created.id.rsplit('_', 1)[0]}_{uuid.uuid4().hex}"
            )
        with pytest.raises(BadRequestError) as streaming:
            await client.responses.retrieve(created.id, stream=True)
        # As in OpenAI, a cursor without a stream changes nothing.
        with_cursor = await client.responses.retrieve(created.id, starting_after=1)

    assert unknown.value.code == "response_not_found"
    assert streaming.value.param == "stream"
    assert with_cursor.id == created.id


def _streaming_config() -> GraphConfig:
    """Return a graph that reports a status and streams a model's answer."""
    model = FakeListChatModel(responses=["Report ready."])

    async def generate(state: MessageState):
        get_stream_writer()(status_event("Searching the web"))
        return {"messages": [await model.ainvoke(state["messages"])]}

    return GraphConfig(
        graph=(
            StateGraph(MessageState)
            .add_node("generate", generate)
            .set_entry_point("generate")
            .set_finish_point("generate")
            .compile()
        ),
        description="Streams in the background",
        features={GraphFeature.BACKGROUND},
    )


async def test_background_stream_delivers_and_replays_the_whole_lifecycle() -> None:
    registry = GraphRegistry(graphs={MODEL: _streaming_config()})
    async with _client(registry) as client:
        stream = await client.responses.create(
            model=MODEL, input="Hello", background=True, stream=True
        )
        events = [event async for event in stream]
        replayed = await client.responses.retrieve(events[0].response.id, stream=True)
        replayed_events = [event async for event in replayed]
        resumed = await client.responses.retrieve(
            events[0].response.id, stream=True, starting_after=3
        )
        resumed_events = [event async for event in resumed]
        retrieved = await client.responses.retrieve(events[0].response.id)

    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [(event.type, event.response.status) for event in events[:3]] == [
        ("response.created", "queued"),
        ("response.queued", "queued"),
        ("response.in_progress", "in_progress"),
    ]
    commentary = [
        event.text
        for event in events
        if event.type == "response.output_text.done"
        and event.text == "Searching the web"
    ]
    assert commentary == ["Searching the web"]
    answer = [
        event.delta
        for event in events
        if event.type == "response.output_text.delta"
        and event.delta != "Searching the web"
    ]
    # The fake model streams one character at a time.
    assert len(answer) > 1
    assert "".join(answer) == "Report ready."
    assert events[-1].type == "response.completed"
    assert events[-1].response == retrieved
    assert replayed_events == events
    assert resumed_events == events[4:]


async def test_background_stream_ends_with_the_failed_response() -> None:
    graph = _Graph(error=OSError("disk full"))
    async with _client(GraphRegistry(graphs={MODEL: graph.config()})) as client:
        stream = await client.responses.create(
            model=MODEL, input="Hello", background=True, stream=True
        )
        events = [event async for event in stream]
        retrieved = await client.responses.retrieve(events[0].response.id)

    terminal = events[-1]
    assert terminal.type == "response.failed"
    assert terminal.response == retrieved
    assert terminal.sequence_number == events[-2].sequence_number + 1
    # Internal exception text never reaches the client.
    assert "disk full" not in retrieved.error.message


class _EventsUnavailable(InMemoryBackgroundBackend):
    """An engine whose event store refuses reads, as without durable streams."""

    async def events(self, run_id: str) -> AsyncGenerator[str, None]:
        for _ in ():
            yield run_id


async def test_background_stream_delivers_the_result_without_stored_events() -> None:
    registry = GraphRegistry(graphs={MODEL: _streaming_config()})
    async with _client(registry, backend=_EventsUnavailable(registry)) as client:
        stream = await client.responses.create(
            model=MODEL, input="Hello", background=True, stream=True
        )
        events = [event async for event in stream]

    assert [event.type for event in events] == [
        "response.created",
        "response.queued",
        "response.completed",
    ]
    assert events[-1].sequence_number == len(events) - 1
    assert events[-1].response.output[-1].content[0].text == "Report ready."


@asynccontextmanager
async def _held_stream(
    graph: _Graph,
) -> AsyncGenerator[tuple[InMemoryBackgroundBackend, BackgroundRun], None]:
    """
    Yield a backend and its streaming run, held inside its graph.

    Yields:
        The backend and the run it is executing.

    """
    registry = GraphRegistry(graphs={MODEL: graph.config()})
    backend = InMemoryBackgroundBackend(registry)
    async with backend.lifespan(None):
        run = await backend.submit(
            BackgroundJob(
                request={
                    "model": MODEL,
                    "input": "Hello",
                    "background": True,
                    "stream": True,
                },
                owner_scope="tenant-a",
                run_id=str(uuid.uuid4()),
                created_at=0,
                idempotency_key="job",
            )
        )
        with fail_after(2):
            await graph.started.wait()
        yield backend, run


async def test_cancelling_a_run_ends_its_stream_with_the_cancelled_response() -> None:
    graph = _Graph(release=Event())
    async with _held_stream(graph) as (backend, run):
        frames = follow_background_response(run, backend)
        first = await anext(frames)
        await backend.cancel(run.id)
        with fail_after(3):
            rest = [frame async for frame in frames]

    assert first.startswith("event: response.created\n")
    assert rest[-1].startswith("event: response.failed\n")
    assert json.loads(rest[-1].partition("data: ")[2])["response"]["status"] == (
        "cancelled"
    )
    assert graph.stopped.is_set()


async def test_leaving_a_stream_leaves_its_run_running() -> None:
    graph = _Graph(release=Event())
    async with _held_stream(graph) as (backend, run):
        frames = follow_background_response(run, backend)
        await anext(frames)
        await frames.aclose()

        graph.release.set()
        with fail_after(2):
            while (current := await backend.get(run.id)).status == "in_progress":
                await checkpoint()

    assert current.status == "completed"
    assert current.snapshot().output_text == "done:Hello"


class _LaggingEvents(InMemoryBackgroundBackend):
    """An engine whose events reach a reader only after the run reads as finished."""

    def __init__(self, registry: GraphRegistry) -> None:
        super().__init__(registry)
        self.finished_read = Event()

    async def get(self, run_id: str) -> BackgroundRun | None:
        run = await super().get(run_id)
        if run is not None and run.status == "completed":
            self.finished_read.set()
        return run

    async def events(self, run_id: str) -> AsyncGenerator[str, None]:
        await self.finished_read.wait()
        async for event in super().events(run_id):
            yield event


async def test_background_stream_delivers_events_that_outlast_their_run() -> None:
    registry = GraphRegistry(graphs={MODEL: _streaming_config()})
    async with _client(registry, backend=_LaggingEvents(registry)) as client:
        stream = await client.responses.create(
            model=MODEL, input="Hello", background=True, stream=True
        )
        events = [event async for event in stream]
        replayed = await client.responses.retrieve(events[0].response.id, stream=True)
        replayed_events = [event async for event in replayed]

    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert "".join(
        event.delta
        for event in events
        if event.type == "response.output_text.delta"
        and event.delta != "Searching the web"
    ) == ("Report ready.")
    # Live and replayed streams number the same log the same way.
    assert replayed_events == events


class _LostEvent(InMemoryBackgroundBackend):
    """An engine whose log misses one event of each run."""

    async def events(self, run_id: str) -> AsyncGenerator[str, None]:
        async for event in super().events(run_id):
            if json.loads(event)["sequence_number"] != LOST_EVENT:
                yield event


async def test_background_stream_never_delivers_an_event_past_a_gap() -> None:
    registry = GraphRegistry(graphs={MODEL: _streaming_config()})
    async with _client(registry, backend=_LostEvent(registry)) as client:
        stream = await client.responses.create(
            model=MODEL, input="Hello", background=True, stream=True
        )
        events = [event async for event in stream]

    # Clients assemble items from these events, so the stream stops at the gap and
    # ends with the whole Response.
    assert [event.sequence_number for event in events] == list(range(LOST_EVENT + 1))
    assert events[-1].type == "response.completed"
    assert events[-1].response.output[-1].content[0].text == "Report ready."


class _StatusBlip(_EventsUnavailable):
    """An engine whose status read fails once while its events are unavailable."""

    def __init__(self, registry: GraphRegistry) -> None:
        super().__init__(registry)
        self.blipped = False

    async def get(self, run_id: str) -> BackgroundRun | None:
        if not self.blipped:
            self.blipped = True
            message = "engine blip"
            raise ConnectionError(message)
        return await super().get(run_id)


async def test_background_stream_outlasts_a_failed_status_check() -> None:
    registry = GraphRegistry(graphs={MODEL: _streaming_config()})
    backend = _StatusBlip(registry)
    async with _client(registry, backend=backend) as client:
        stream = await client.responses.create(
            model=MODEL, input="Hello", background=True, stream=True
        )
        events = [event async for event in stream]

    assert backend.blipped
    assert events[-1].type == "response.completed"


@pytest.mark.parametrize(
    ("finish_reason", "refusal"),
    [
        pytest.param(None, "Cannot cancel a completed response.", id="completed"),
        pytest.param(
            "length", "Cannot cancel an incomplete response.", id="incomplete"
        ),
    ],
)
async def test_cancellation_stops_an_active_run_and_refuses_a_finished_one(
    finish_reason: str | None,
    refusal: str,
) -> None:
    graph = _Graph(release=Event(), finish_reason=finish_reason)
    async with _client(GraphRegistry(graphs={MODEL: graph.config()})) as client:
        active = await client.responses.create(
            model=MODEL, input="Hello", background=True
        )
        with fail_after(2):
            await graph.started.wait()
        cancelled = await client.responses.cancel(active.id)
        with fail_after(2):
            await graph.stopped.wait()
        after_cancel = await client.responses.retrieve(active.id)
        cancelled_again = await client.responses.cancel(active.id)
        graph.release.set()
        finished = await client.responses.create(
            model=MODEL, input="Again", background=True
        )
        completed = await _finished(client, finished.id)
        with pytest.raises(BadRequestError) as refused:
            await client.responses.cancel(finished.id)
        after_refusal = await client.responses.retrieve(finished.id)

    assert cancelled.status == after_cancel.status == cancelled_again.status
    assert cancelled.status == "cancelled"
    # OpenAI's own words for a Response that already finished.
    assert refused.value.body["message"] == refusal
    assert after_refusal == completed


async def test_unexpected_graph_failure_fails_the_response() -> None:
    graph = _Graph(error=OSError("disk full"))
    async with _client(GraphRegistry(graphs={MODEL: graph.config()})) as client:
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
    registry = GraphRegistry(
        graphs={MODEL: graph.config(), "other-model": graph.config()}
    )
    async with _client(registry) as client:
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
        with pytest.raises(UnprocessableEntityError) as other_model:
            await client.responses.create(
                model="other-model",
                input="Hello",
                background=True,
                extra_headers=headers,
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
    assert reused.value.code == other_model.value.code == "idempotency_key_reused"
    assert too_long.value.param == "Idempotency-Key"
    assert graph.inputs == ["Hello", "Hello"]


async def test_one_of_two_answers_to_a_pause_continues_the_run(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    registry = GraphRegistry(
        graphs={
            "approval": GraphConfig(
                graph=make_multi_interrupt_graph(sqlite_checkpointer),
                description="Two questions",
                features={GraphFeature.INTERRUPTS, GraphFeature.BACKGROUND},
                request_to_input=lambda _request, _messages: {"answers": []},
                output_to_message=lambda output: AIMessage(
                    content=",".join(output["answers"])
                ),
            )
        },
        run_coordinator=InMemoryRunCoordinator(),
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


async def test_a_job_that_runs_again_starts_over(
    sqlite_checkpointer: AsyncSqliteSaver,
) -> None:
    def review(state: MessageState):
        answer = interrupt({"question": "Approve?"})
        inputs = ",".join(m.text for m in state["messages"] if m.type == "human")
        return {"messages": [AIMessage(content=f"{answer}:{inputs}")]}

    graph = (
        StateGraph(MessageState)
        .add_node("review", review)
        .set_entry_point("review")
        .set_finish_point("review")
        .compile(checkpointer=sqlite_checkpointer)
    )
    registry = GraphRegistry(
        graphs={
            "review": GraphConfig(
                graph=graph,
                description="One question",
                features={GraphFeature.INTERRUPTS, GraphFeature.BACKGROUND},
            )
        },
        run_coordinator=InMemoryRunCoordinator(),
    )
    job = BackgroundJob(
        request={"model": "review", "input": "Hi", "background": True},
        owner_scope="tenant-a",
        run_id=str(uuid.uuid4()),
        created_at=0,
        idempotency_key="job",
    )
    engine_run_id = str(uuid.uuid4())

    await execute_background_job(job, engine_run_id, registry)
    # The worker died before the engine recorded the result, so it runs again.
    paused = Response.model_validate(
        await execute_background_job(job, engine_run_id, registry)
    )
    async with _client(registry) as client:
        final = await client.responses.create(
            model="review",
            previous_response_id=paused.id,
            input=resume_outputs(paused, ["approve"]),
        )

    assert final.output_text == "approve:Hi"
