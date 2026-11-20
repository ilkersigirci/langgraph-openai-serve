"""Create, retrieve, follow, and cancel background Responses."""

import hashlib
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing

from anyio import EndOfStream, move_on_after, sleep_forever
from anyio.streams.memory import MemoryObjectReceiveStream
from openai.types.responses import Response
from pydantic import BaseModel

from langgraph_openai_serve.api.responses.events import (
    BACKGROUND_ACCEPTED_EVENTS,
    background_accepted_events,
    encode_event,
    terminal_event,
)
from langgraph_openai_serve.api.responses.interrupts import interrupt_response_nonce
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background import (
    BackgroundBackend,
    BackgroundJob,
    BackgroundRun,
)
from langgraph_openai_serve.core.errors import InvalidRequestError
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.runner import TaskStream

logger = get_logger(__name__)

# 255 matches Stripe's Idempotency-Key limit.
_MAX_IDEMPOTENCY_KEY_LENGTH = 255

# How long a quiet stream waits before asking the engine whether the run ended.
_RUN_CHECK_SECONDS = 1.0

_TERMINAL_EVENT_TYPES = {
    "response.completed",
    "response.incomplete",
    "response.failed",
}


class _PublishedEvent(BaseModel):
    """The fields of a stored event that decide its delivery."""

    type: str
    sequence_number: int


async def create_background_response(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
    idempotency_key: str | None = None,
) -> Response | AsyncGenerator[str, None]:
    """
    Validate one request and submit it to the background engine.

    Proxies and SDKs retry creates after timeouts and 5xx responses. With an
    ``Idempotency-Key``, a retry finds the original run instead of starting a
    second one.

    Returns:
        The accepted Response, or its event stream for ``stream=true``.

    """
    if background is None:
        message = "Background Responses are not configured for this server."
        raise InvalidRequestError(message, param="background")
    job = _job(
        request,
        graph_registry,
        owner_scope=checkpoint_scope,
        idempotency_key=idempotency_key,
    )
    run = await background.submit(job)
    if run.job.request != job.request:
        # 422 follows the IETF Idempotency-Key draft; SDKs do not retry it.
        message = "This Idempotency-Key was already used with a different request."
        raise InvalidRequestError(
            message,
            param="Idempotency-Key",
            code="idempotency_key_reused",
            status_code=422,
        )
    if request.stream:
        return follow_background_response(run, background)
    return run.snapshot()


async def retrieve_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
    stream: bool = False,
    starting_after: int | None = None,
) -> Response | AsyncGenerator[str, None]:
    """
    Read one authorized Response.

    Returns:
        Its current snapshot or, with ``stream``, its event stream.

    """
    run = await _visible_run(response_id, background, checkpoint_scope)
    if not stream:
        return run.snapshot()
    # As in OpenAI: only a job created to stream has published its events.
    if background is None or not run.job.request.stream:
        message = "This Response was not created with stream=true."
        raise InvalidRequestError(message, param="stream")
    return follow_background_response(run, background, starting_after=starting_after)


async def follow_background_response(
    run: BackgroundRun,
    background: BackgroundBackend,
    *,
    starting_after: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Replay a Response's published events and follow them until it ends.

    The events are progress and the run's status is the truth. When the engine
    reports a finished run first, because the run was cancelled, its worker died,
    or its events could not be stored, the stored Response ends the stream
    instead.

    Yields:
        Named, compact Responses SSE frames.

    """
    cursor = -1 if starting_after is None else starting_after
    for event in background_accepted_events(run.queued_snapshot()):
        if event.sequence_number > cursor:
            yield encode_event(event)

    async with aclosing(_published_frames(run.id, background, cursor)) as frames:
        async for frame in frames:
            yield frame


async def _published_frames(
    run_id: str, background: BackgroundBackend, cursor: int
) -> AsyncGenerator[str, None]:
    """
    Follow the contiguous worker log, using the stored result if it stops early.

    Yields:
        SSE frames after the client's cursor.

    """
    next_sequence = BACKGROUND_ACCEPTED_EVENTS
    finished = False
    async with aclosing(
        TaskStream(background.events(run_id), name="background-events")
    ) as published:
        while True:
            payload = await _next_event(published.receive)
            if payload is not None:
                event = _PublishedEvent.model_validate_json(payload)
                # Clients assemble deltas in order. After a gap, only the stored
                # terminal Response can supply a complete result.
                if event.sequence_number != next_sequence:
                    continue
                next_sequence += 1
                if event.sequence_number > cursor:
                    yield f"event: {event.type}\ndata: {payload}\n\n"
                if event.type in _TERMINAL_EVENT_TYPES:
                    return
                continue

            try:
                current = await background.get(run_id)
            except Exception:
                logger.warning("background.status_unavailable", exc_info=True)
                continue
            if current is None:
                return
            if current.status in {"queued", "in_progress"}:
                continue
            # Stored events can arrive after the engine reports completion.
            # Give them one further quiet interval before closing the stream.
            if finished:
                # The client may have seen events we can no longer read.
                yield encode_event(
                    terminal_event(current.snapshot(), max(next_sequence, cursor + 1))
                )
                return
            finished = True


async def _next_event(published: MemoryObjectReceiveStream[str]) -> str | None:
    """Return the next published event, or None when it is time to check the run."""
    with move_on_after(_RUN_CHECK_SECONDS):
        try:
            return await published.receive()
        except EndOfStream:
            # No more events will come, so only the run's status can end the stream.
            await sleep_forever()
    return None


async def cancel_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Cancel an active Response, as often as needed; a finished one cannot be."""
    run = await _visible_run(response_id, background, checkpoint_scope)
    if run.status in {"completed", "failed"}:
        status = run.snapshot().status
        # OpenAI's words, such as "Cannot cancel an incomplete response."
        article = "an" if status == "incomplete" else "a"
        message = f"Cannot cancel {article} {status} response."
        raise InvalidRequestError(message)
    if background is not None and run.status in {"queued", "in_progress"}:
        await background.cancel(run.id)
        # A run that finished before the cancel keeps its outcome; one that
        # still reads as active is being cancelled.
        run = await background.get(run.id) or run
        if run.status in {"queued", "in_progress"}:
            run = run.model_copy(update={"status": "cancelled"})
    return run.snapshot()


def _job(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    *,
    owner_scope: str,
    idempotency_key: str | None,
) -> BackgroundJob:
    """Validate the request as the worker will decode it."""
    graph_config = graph_registry.get_graph(request.model)
    graph_request, _messages, resume = decode_responses_request(request, graph_config)
    # The worker validates settings only while it builds the run's context, so
    # check them here to reject the request instead of queuing a failing job.
    if graph_config.client_settings is not None:
        graph_config.client_settings.validate_request(graph_request)
    return BackgroundJob(
        request=request,
        owner_scope=owner_scope,
        # An answer continues its paused run's checkpoint thread.
        run_id=resume.run_id if resume is not None else str(uuid.uuid4()),
        created_at=int(time.time()),
        idempotency_key=(
            _scoped_idempotency_key(owner_scope, idempotency_key)
            if idempotency_key is not None
            else uuid.uuid4().hex
        ),
    )


def _scoped_idempotency_key(owner_scope: str, idempotency_key: str) -> str:
    """Digest the key with its owner so the engine never sees the client value."""
    if not 0 < len(idempotency_key) <= _MAX_IDEMPOTENCY_KEY_LENGTH:
        message = (
            f"Idempotency-Key must be 1 to {_MAX_IDEMPOTENCY_KEY_LENGTH} characters."
        )
        raise InvalidRequestError(message, param="Idempotency-Key")
    identity = json.dumps([owner_scope, idempotency_key], separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()


async def _visible_run(
    response_id: str,
    background: BackgroundBackend | None,
    checkpoint_scope: str,
) -> BackgroundRun:
    """Return a run only to its owner; every other case is indistinguishable."""
    run_id = interrupt_response_nonce(response_id)
    run = (
        await background.get(run_id)
        if background is not None and run_id is not None
        else None
    )
    if (
        run is None
        or run.response_id != response_id
        or run.job.owner_scope != checkpoint_scope
    ):
        msg = f"Response '{response_id}' was not found."
        raise InvalidRequestError(
            msg,
            param="response_id",
            code="response_not_found",
            status_code=404,
        )
    return run
