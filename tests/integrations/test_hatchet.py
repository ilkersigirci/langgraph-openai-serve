import base64
import json
import uuid
from collections.abc import Iterable
from datetime import timedelta
from types import SimpleNamespace

import grpc
import pytest
from hatchet_sdk import (
    ClientConfig,
    Hatchet,
    IdempotencyCollisionError,
    RunStatus,
    V1TaskStatus,
)
from hatchet_sdk.clients.admin import TaskRunDetail, WorkflowRunDetail
from tests.graph.support.message import make_message_graph

from langgraph_openai_serve import (
    BackgroundJob,
    GraphConfig,
    GraphFeature,
    GraphRegistry,
)
from langgraph_openai_serve.api.responses.events import BACKGROUND_ACCEPTED_EVENTS
from langgraph_openai_serve.integrations.hatchet import (
    HatchetBackgroundBackend,
    create_hatchet_task,
)

RUN_ID = str(uuid.uuid4())
JOB = BackgroundJob(
    request={"model": "background", "input": "Hello", "background": True},
    owner_scope="tenant-a",
    run_id=str(uuid.uuid4()),
    created_at=1_700_000_000,
    idempotency_key="key",
)
STREAMING_JOB = JOB.model_copy(
    update={"request": JOB.request.model_copy(update={"stream": True})}
)


def _rpc_error(code: grpc.StatusCode) -> grpc.aio.AioRpcError:
    metadata = grpc.aio.Metadata()
    return grpc.aio.AioRpcError(code, metadata, metadata)


class _Streams:
    """A durable stream store whose calls raise the given errors first, in order."""

    def __init__(
        self,
        *,
        published: tuple[str, ...] = (),
        publish_errors: Iterable[Exception] = (),
        metadata_errors: Iterable[Exception] = (),
    ) -> None:
        self.published = list(published)
        self.batch_timeouts: list[timedelta | None] = []
        self.topics: set[str] = set()
        self.publish_attempts = 0
        self.publish_errors = iter(publish_errors)
        self.metadata_errors = iter(metadata_errors)
        self.metadata_reads = 0

    async def publish(
        self, topic: str, message: str, batch_timeout: timedelta | None = None
    ) -> None:
        self.publish_attempts += 1
        if (error := next(self.publish_errors, None)) is not None:
            raise error
        self.topics.add(topic)
        self.published.append(message)
        self.batch_timeouts.append(batch_timeout)

    async def aio_topic_metadata(self, topic: str) -> object:
        self.metadata_reads += 1
        if (error := next(self.metadata_errors, None)) is not None:
            raise error
        if not self.published:
            raise _rpc_error(grpc.StatusCode.NOT_FOUND)
        return SimpleNamespace(topic=topic)


class _Task:
    """Captures the options and function the adapter registers."""

    def __init__(
        self, error: Exception | None = None, streams: _Streams | None = None
    ) -> None:
        self.options: dict[str, object] = {}
        self.error = error
        self.streams = streams or _Streams()

    def task(self, **options):
        self.options = options

        def register(function):
            self.run = function
            return self

        return register

    async def aio_run(self, job: BackgroundJob, *, wait_for_result: bool):
        assert wait_for_result is False
        if self.error is not None:
            raise self.error
        return SimpleNamespace(workflow_run_id=RUN_ID)


class _Runs:
    def __init__(self, *details: WorkflowRunDetail) -> None:
        self.details = {detail.external_id: detail for detail in details}
        self.cancelled: list[str] = []

    async def aio_get_details(self, run_id: str) -> WorkflowRunDetail:
        if run_id not in self.details:
            raise _rpc_error(grpc.StatusCode.NOT_FOUND)
        return self.details[run_id]

    async def aio_cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)


def _completed(response: dict[str, object]) -> WorkflowRunDetail:
    return WorkflowRunDetail(
        external_id=RUN_ID,
        status=RunStatus.COMPLETED,
        input=JOB.model_dump(mode="json"),
        task_runs={
            "lgos-background-response": TaskRunDetail(
                external_id=str(uuid.uuid4()),
                readable_id="lgos-background-response",
                output=response,
                status=V1TaskStatus.COMPLETED,
            )
        },
        done=True,
    )


def test_task_registers_with_the_pinned_hatchet_sdk() -> None:
    claims = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "00000000-0000-0000-0000-000000000000",
                "server_url": "http://localhost:8080",
                "grpc_broadcast_address": "localhost:7070",
            }
        ).encode()
    )
    token = f"eyJhbGciOiJub25lIn0.{claims.decode().rstrip('=')}.signature"

    task = create_hatchet_task(Hatchet(config=ClientConfig(token=token)))

    assert task.name == "lgos-background-response"


async def test_task_executes_the_job_with_the_worker_lifespan_graphs() -> None:
    hatchet = _Task()
    create_hatchet_task(hatchet)  # ty: ignore[invalid-argument-type] - Captures the native registration.
    registry = GraphRegistry(
        graphs={
            "background": GraphConfig(
                graph=make_message_graph("report"),
                description="Background test graph",
                features={GraphFeature.BACKGROUND},
            )
        }
    )
    context = SimpleNamespace(lifespan=registry, workflow_run_id=RUN_ID)

    response = await hatchet.run(JOB, context)

    # The key must name the job field that carries the scoped Idempotency-Key.
    assert hatchet.options["idempotency"].key_expression == "input.idempotency_key"
    assert response["status"] == "completed"
    assert response["id"].endswith(uuid.UUID(RUN_ID).hex)
    assert response["output"][0]["content"][0]["text"] == "report"
    # A job that was not created to stream writes nothing to the engine.
    assert hatchet.streams.publish_attempts == 0
    assert hatchet.streams.metadata_reads == 0
    with pytest.raises(TypeError, match="GraphRegistry"):
        await hatchet.run(JOB, SimpleNamespace(lifespan=None, workflow_run_id=RUN_ID))


def _streaming_task(streams: _Streams) -> tuple[_Task, SimpleNamespace]:
    hatchet = _Task(streams=streams)
    create_hatchet_task(hatchet)  # ty: ignore[invalid-argument-type] - Captures the native registration.
    registry = GraphRegistry(
        graphs={
            "background": GraphConfig(
                graph=make_message_graph("report"),
                description="Background test graph",
                features={GraphFeature.BACKGROUND},
            )
        }
    )
    return hatchet, SimpleNamespace(lifespan=registry, workflow_run_id=RUN_ID)


async def test_streaming_job_publishes_its_events_to_the_runs_topic() -> None:
    streams = _Streams()
    hatchet, context = _streaming_task(streams)

    response = await hatchet.run(STREAMING_JOB, context)

    assert streams.topics == {f"lgos-response-{RUN_ID}"}
    events = [json.loads(event) for event in streams.published]
    # The API already sent the events that accepted the Response.
    assert events[0]["type"] == "response.in_progress"
    assert [event["sequence_number"] for event in events] == list(
        range(BACKGROUND_ACCEPTED_EVENTS, BACKGROUND_ACCEPTED_EVENTS + len(events))
    )
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"] == response
    deltas = [
        (event["delta"], timeout)
        for event, timeout in zip(events, streams.batch_timeouts, strict=True)
        if event["type"] == "response.output_text.delta"
    ]
    # Only text deltas may wait to share a publish request.
    assert "".join(delta for delta, _ in deltas) == "report"
    assert all(timeout is not None for _, timeout in deltas)
    assert streams.batch_timeouts.count(None) == len(events) - len(deltas)


async def test_rerun_of_a_streaming_job_keeps_the_first_attempts_events() -> None:
    first_attempt = ('{"type":"response.in_progress","sequence_number":2}',)
    streams = _Streams(published=first_attempt)
    hatchet, context = _streaming_task(streams)

    response = await hatchet.run(STREAMING_JOB, context)

    assert response["status"] == "completed"
    assert tuple(streams.published) == first_attempt
    assert streams.publish_attempts == 0


UNAVAILABLE = _rpc_error(grpc.StatusCode.UNAVAILABLE)


@pytest.mark.parametrize(
    ("streams", "attempts"),
    [
        # Only an engine that reports no topic proves this attempt is the first.
        pytest.param(
            _Streams(metadata_errors=[_rpc_error(grpc.StatusCode.PERMISSION_DENIED)]),
            0,
            id="no-durable-streams",
        ),
        pytest.param(_Streams(metadata_errors=[UNAVAILABLE]), 0, id="unreadable-topic"),
        pytest.param(_Streams(publish_errors=[UNAVAILABLE]), 1, id="publish-refused"),
    ],
)
async def test_streaming_job_completes_when_its_events_cannot_be_stored(
    streams: _Streams,
    attempts: int,
) -> None:
    hatchet, context = _streaming_task(streams)

    response = await hatchet.run(STREAMING_JOB, context)

    assert response["status"] == "completed"
    assert response["output"][0]["content"][0]["text"] == "report"
    # The first failure ends publishing.
    assert streams.publish_attempts == attempts
    assert streams.published == []


async def test_backend_submits_reads_and_cancels_hatchet_runs() -> None:
    runs = _Runs(_completed({"id": "resp_done", "status": "completed"}))
    backend = HatchetBackgroundBackend(_Task(), runs, _Streams())  # ty: ignore[invalid-argument-type] - Fakes of the native task and clients.

    submitted = await backend.submit(JOB)
    completed = await backend.get(RUN_ID)
    missing = await backend.get(str(uuid.uuid4()))
    await backend.cancel(RUN_ID)

    assert (submitted.id, submitted.status) == (RUN_ID, "queued")
    assert completed is not None
    assert completed.job == JOB
    assert completed.response == {"id": "resp_done", "status": "completed"}
    assert missing is None
    assert runs.cancelled == [RUN_ID]


async def test_idempotency_collision_returns_the_existing_run() -> None:
    task = _Task(error=IdempotencyCollisionError(RUN_ID))
    backend = HatchetBackgroundBackend(
        task,  # ty: ignore[invalid-argument-type] - Fake of the native task.
        _Runs(_completed({"id": "resp_done", "status": "completed"})),  # ty: ignore[invalid-argument-type] - Fake of the runs client.
        _Streams(),  # ty: ignore[invalid-argument-type] - Fake of the streams client.
    )

    existing = await backend.submit(JOB)

    assert existing.status == "completed"
    assert existing.response == {"id": "resp_done", "status": "completed"}


class _Subscriptions:
    """Serves each subscribe call from one script of events or an error."""

    def __init__(self, *scripts: list[str] | Exception) -> None:
        self.scripts = list(scripts)
        self.cursors: list[str | None] = []

    async def subscribe(self, topic: str, cursor: str | None = None):
        assert topic == f"lgos-response-{RUN_ID}"
        self.cursors.append(cursor)
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        for payload in script:
            yield SimpleNamespace(payload=payload.encode(), cursor=f"after-{payload}")


def _backend(streams: _Subscriptions) -> HatchetBackgroundBackend:
    return HatchetBackgroundBackend(_Task(), _Runs(), streams)  # ty: ignore[invalid-argument-type] - Fakes of the native task and clients.


async def test_events_resume_from_the_last_event_after_an_idle_hangup() -> None:
    # A subscription that ends without an error is the engine's idle hangup.
    streams = _Subscriptions(
        ["a", "b"], ["c"], _rpc_error(grpc.StatusCode.PERMISSION_DENIED)
    )

    events = [event async for event in _backend(streams).events(RUN_ID)]

    assert events == ["a", "b", "c"]
    assert streams.cursors == [None, "after-b", "after-c"]


async def test_events_end_when_the_tenant_has_no_durable_streams() -> None:
    streams = _Subscriptions(_rpc_error(grpc.StatusCode.PERMISSION_DENIED))

    assert [event async for event in _backend(streams).events(RUN_ID)] == []
