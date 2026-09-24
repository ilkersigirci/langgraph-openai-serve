"""Create, retrieve, and cancel polling-only background Responses."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from anyio import CancelScope
from langchain_core.runnables import RunnableConfig
from openai.types.responses import Response

from langgraph_openai_serve.api.responses.interrupts import interrupt_response_id
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_graph_request,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background.contracts import BackgroundBackend
from langgraph_openai_serve.background.responses import (
    cancelled_response,
    finish_response,
    queued_response,
    response_json,
    transcript_ids,
)
from langgraph_openai_serve.background.store import NewRun, ResponseStatus, StoredRun
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig, GraphRegistry
from langgraph_openai_serve.graph.interrupt.models import InterruptResume
from langgraph_openai_serve.graph.interrupt.state import (
    checkpoint_key,
    normalize_checkpoint_scope,
    prepare_interrupt_state,
)
from langgraph_openai_serve.graph.runner import root_messages

if TYPE_CHECKING:
    from pydantic import JsonValue

logger = get_logger(__name__)

# Stripe and the IETF Idempotency-Key draft both bound keys; 255 matches Stripe.
_MAX_IDEMPOTENCY_KEY_LENGTH = 255


class BackgroundResponseNotFoundError(LookupError):
    """Raised for unknown, expired, or unauthorized background Response IDs."""


class IdempotencyKeyReusedError(ValueError):
    """Raised when an Idempotency-Key is reused with a different request."""


async def accept_background_response(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
    idempotency_key: str | None = None,
) -> Response:
    """
    Validate, persist, and submit one polling-only background Response.

    Proxies and SDKs retry creates after timeouts and 5xx responses. With an
    ``Idempotency-Key``, a retry returns the original Response instead of
    starting a second run.
    """
    if background is None:
        message = "Background Responses are not configured for this server."
        raise UnsupportedResponsesRequestError(message, param="background")
    owner_scope = normalize_checkpoint_scope(checkpoint_scope)
    digest, fingerprint = _idempotency(request, owner_scope, idempotency_key)
    # Like Stripe, return a retried request's first result before validation:
    # an accepted interrupt answer changes the state it was validated against.
    if digest is not None and (existing := await background.store.find(digest)):
        return _replayed(existing, fingerprint)
    new_run = await _new_run(
        request,
        graph_registry,
        owner_scope=owner_scope,
        idempotency_digest=digest,
        request_fingerprint=fingerprint,
    )
    # A cancelled request must not leave a persisted run unsubmitted.
    with CancelScope(shield=True):
        run = await background.store.create(new_run)
        if run.response_id == new_run.response_id:
            await _submit(background, run)
    return _replayed(run, fingerprint)


async def retrieve_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Read one authorized snapshot without scheduling side effects."""
    if background is None:
        raise BackgroundResponseNotFoundError(response_id)
    run = await _visible_run(response_id, background, checkpoint_scope)
    return Response.model_validate(run.response)


async def cancel_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Cancel an active Response; a terminal one is returned unchanged."""
    if background is None:
        raise BackgroundResponseNotFoundError(response_id)
    run = await _visible_run(response_id, background, checkpoint_scope)
    if run.terminal:
        return Response.model_validate(run.response)
    # Commit the public outcome first; a run that keeps executing afterwards
    # cannot overwrite it, so stopping the work is best effort.
    winner = await finish_response(
        background.store,
        background.settings,
        run.response_id,
        cancelled_response(Response.model_validate(run.response)),
    )
    if winner is None:
        raise BackgroundResponseNotFoundError(response_id)
    if winner.status is ResponseStatus.CANCELLED:
        try:
            await background.stop(winner)
        except Exception:
            logger.exception(
                "background.stop_failed",
                extra={"response_id": response_id},
            )
    return Response.model_validate(winner.response)


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
    return (
        _json_digest([owner_scope, request.model, idempotency_key]),
        _background_fingerprint(request),
    )


def _replayed(run: StoredRun, request_fingerprint: str | None) -> Response:
    if run.request_fingerprint != request_fingerprint:
        raise IdempotencyKeyReusedError
    return Response.model_validate(run.response)


async def _new_run(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    *,
    owner_scope: str,
    idempotency_digest: str | None,
    request_fingerprint: str | None,
) -> NewRun:
    graph_config = graph_registry.get_graph(request.model)
    if not graph_config.supports(GraphFeature.BACKGROUND):
        message = f"Model '{request.model}' does not support background execution."
        raise UnsupportedResponsesRequestError(message, param="background")
    graph_request, messages, resume = decode_graph_request(request, graph_config)
    if graph_config.client_settings is not None:
        graph_config.client_settings.validate_request(graph_request)
    if "lgos_run_id" in graph_request.metadata:
        message = (
            "metadata.lgos_run_id is only supported for interrupt-enabled "
            "foreground Responses."
        )
        raise UnsupportedResponsesRequestError(
            message,
            param="metadata.lgos_run_id",
        )

    # Every Response of one run shares the foreground interrupt identity and
    # checkpoint thread, so an answer in either mode continues the paused run.
    run_id = resume.run_id if resume is not None else str(uuid.uuid4())
    response_id = interrupt_response_id(run_id)
    checkpoint_thread_id = checkpoint_key(request.model, run_id, scope=owner_scope)
    prior_ids = (
        await _paused_transcript_ids(graph_config, checkpoint_thread_id, resume)
        if resume is not None
        else transcript_ids(messages)
    )
    now = datetime.now(UTC)
    return NewRun(
        response_id=response_id,
        owner_scope=owner_scope,
        model=request.model,
        checkpoint_thread_id=checkpoint_thread_id,
        envelope=cast(
            "dict[str, JsonValue]",
            request.model_dump(mode="json", by_alias=True),
        ),
        response=response_json(
            queued_response(
                request,
                response_id=response_id,
                created_at=int(now.timestamp()),
            )
        ),
        created_at=now,
        prior_ids=prior_ids,
        idempotency_digest=idempotency_digest,
        request_fingerprint=request_fingerprint,
    )


async def _paused_transcript_ids(
    graph_config: GraphConfig,
    checkpoint_thread_id: str,
    resume: InterruptResume,
) -> tuple[str, ...]:
    """Validate an answer like a foreground resume; return what precedes it."""
    graph = await graph_config.resolve_graph()
    config = RunnableConfig(configurable={"thread_id": checkpoint_thread_id})
    await prepare_interrupt_state(graph, config, resume.run_id, resume)
    snapshot = await graph.aget_state(config)
    return transcript_ids(root_messages(snapshot.values))


async def _submit(background: BackgroundBackend, run: StoredRun) -> None:
    try:
        await background.submit(run)
    except Exception:
        # The run stays queued; maintenance resubmits it after resubmit_after.
        logger.exception(
            "background.submission_failed",
            extra={"response_id": run.response_id},
        )


async def _visible_run(
    response_id: str,
    background: BackgroundBackend,
    checkpoint_scope: str,
) -> StoredRun:
    run = await background.store.get(response_id)
    owner_scope = normalize_checkpoint_scope(checkpoint_scope)
    if run is None or not run.visible_to(owner_scope, now=datetime.now(UTC)):
        raise BackgroundResponseNotFoundError(response_id)
    return run


def _background_fingerprint(request: ResponseCreateRequest) -> str:
    # Equivalent spellings of the accepted mode must not look like new content.
    normalized = request.model_copy(
        update={"background": True, "stream": False, "store": bool(request.store)}
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    return _json_digest(normalized)


def _json_digest(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "BackgroundResponseNotFoundError",
    "IdempotencyKeyReusedError",
    "accept_background_response",
    "cancel_background_response",
    "retrieve_background_response",
]
