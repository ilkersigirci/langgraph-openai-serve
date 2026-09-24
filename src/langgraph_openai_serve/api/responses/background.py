"""Create, retrieve, and cancel polling-only background Responses."""

import hashlib
import json
import time
import uuid

from openai.types.responses import Response

from langgraph_openai_serve.api.responses.interrupts import interrupt_response_nonce
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_graph_request,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background import (
    BackgroundBackend,
    BackgroundJob,
    BackgroundRun,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt.state import normalize_checkpoint_scope
from langgraph_openai_serve.protocol import RUN_METADATA_KEY

# Stripe and the IETF Idempotency-Key draft both bound keys; 255 matches Stripe.
_MAX_IDEMPOTENCY_KEY_LENGTH = 255


class BackgroundResponseNotFoundError(LookupError):
    """Raised for unknown, expired, or unauthorized background Response IDs."""


class IdempotencyKeyReusedError(ValueError):
    """Raised when an Idempotency-Key is reused with a different request."""


async def create_background_response(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
    idempotency_key: str | None = None,
) -> Response:
    """
    Validate one request and submit it to the background engine.

    Proxies and SDKs retry creates after timeouts and 5xx responses. With an
    ``Idempotency-Key``, a retry returns the original Response instead of
    starting a second run.
    """
    if background is None:
        message = "Background Responses are not configured for this server."
        raise UnsupportedResponsesRequestError(message, param="background")
    job = _job(
        request,
        graph_registry,
        owner_scope=normalize_checkpoint_scope(checkpoint_scope),
        idempotency_key=idempotency_key,
    )
    run = await background.submit(job)
    if run.job.request_fingerprint != job.request_fingerprint:
        raise IdempotencyKeyReusedError
    return run.snapshot()


async def retrieve_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Read one authorized Response snapshot."""
    if background is None:
        raise BackgroundResponseNotFoundError(response_id)
    run = await _visible_run(response_id, background, checkpoint_scope)
    return run.snapshot()


async def cancel_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Cancel an active Response; a finished one is returned unchanged."""
    if background is None:
        raise BackgroundResponseNotFoundError(response_id)
    run = await _visible_run(response_id, background, checkpoint_scope)
    if run.status in {"queued", "in_progress"}:
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
    if not graph_config.supports(GraphFeature.BACKGROUND):
        message = f"Model '{request.model}' does not support background execution."
        raise UnsupportedResponsesRequestError(message, param="background")
    graph_request, _messages, resume = decode_graph_request(request, graph_config)
    if graph_config.client_settings is not None:
        graph_config.client_settings.validate_request(graph_request)
    if RUN_METADATA_KEY in graph_request.metadata:
        message = (
            f"metadata.{RUN_METADATA_KEY} is only supported for interrupt-enabled "
            "foreground Responses."
        )
        raise UnsupportedResponsesRequestError(
            message, param=f"metadata.{RUN_METADATA_KEY}"
        )
    digest, fingerprint = _idempotency(request, owner_scope, idempotency_key)
    return BackgroundJob(
        request=request.model_dump(mode="json", by_alias=True),
        owner_scope=owner_scope,
        # An answer continues its paused run's checkpoint thread.
        run_id=resume.run_id if resume is not None else str(uuid.uuid4()),
        created_at=int(time.time()),
        idempotency_key=digest or uuid.uuid4().hex,
        request_fingerprint=fingerprint,
    )


def _idempotency(
    request: ResponseCreateRequest,
    owner_scope: str,
    idempotency_key: str | None,
) -> tuple[str | None, str | None]:
    """Return the scoped key digest and the request fingerprint it protects."""
    if idempotency_key is None:
        return None, None
    if not 0 < len(idempotency_key) <= _MAX_IDEMPOTENCY_KEY_LENGTH:
        message = (
            f"Idempotency-Key must be 1 to {_MAX_IDEMPOTENCY_KEY_LENGTH} characters."
        )
        raise UnsupportedResponsesRequestError(message, param="Idempotency-Key")
    # Equivalent spellings of the accepted mode must not look like new content.
    normalized = request.model_copy(
        update={"stream": False, "store": bool(request.store)}
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    return (
        _json_digest([owner_scope, request.model, idempotency_key]),
        _json_digest(normalized),
    )


async def _visible_run(
    response_id: str,
    background: BackgroundBackend,
    checkpoint_scope: str,
) -> BackgroundRun:
    run_id = interrupt_response_nonce(response_id)
    run = await background.get(run_id) if run_id is not None else None
    if (
        run is None
        or run.response_id != response_id
        or run.job.owner_scope != normalize_checkpoint_scope(checkpoint_scope)
    ):
        raise BackgroundResponseNotFoundError(response_id)
    return run


def _json_digest(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "BackgroundResponseNotFoundError",
    "IdempotencyKeyReusedError",
    "cancel_background_response",
    "create_background_response",
    "retrieve_background_response",
]
