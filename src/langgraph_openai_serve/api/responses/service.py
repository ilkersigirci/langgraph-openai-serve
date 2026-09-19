"""Prepare and execute graph runs for OpenAI Responses."""

import hashlib
import json
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import CustomStreamPart, UpdatesStreamPart
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)

from langgraph_openai_serve.api.responses.events import (
    ResponsesEventBuilder,
    encode_event,
)
from langgraph_openai_serve.api.responses.output import (
    UnsupportedResponsesOutputError,
    response_usage,
)
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_responses_request,
    selected_server_tools,
    validate_background_request,
    validate_tools,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.background.contracts import BackgroundBackend
from langgraph_openai_serve.background.responses import (
    active_response,
    cancelled_response,
    response_json,
)
from langgraph_openai_serve.background.store import NewRun, ResponseStatus
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.events import parse_status_event
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.interrupt.state import (
    checkpoint_key,
    normalize_checkpoint_scope,
    normalize_run_id,
)
from langgraph_openai_serve.graph.runner import invoke_run, stream_run
from langgraph_openai_serve.graph.utils import GraphRun, prepare_run

logger = get_logger(__name__)

if TYPE_CHECKING:
    from pydantic import JsonValue


class BackgroundResponseNotFoundError(LookupError):
    """Raised for unknown, expired, or unauthorized background Response IDs."""


async def accept_background_response(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Validate and durably accept one polling-only background Response."""
    validate_background_request(request)
    if background is None:
        message = "Background Responses are not configured for this server."
        raise UnsupportedResponsesRequestError(message, param="background")

    graph_config = graph_registry.get_graph(request.model)
    policy = graph_config.background
    if policy is None:
        message = f"Model '{request.model}' does not support background execution."
        raise UnsupportedResponsesRequestError(message, param="background")
    validate_tools(request, graph_config.server_tools)
    graph_request, messages, _ = decode_responses_request(
        request,
        graph_config.server_tools,
    )

    owner_scope = normalize_checkpoint_scope(checkpoint_scope)
    idempotency_key = graph_request.metadata.get("lgos_run_id")
    if idempotency_key is not None:
        idempotency_key = normalize_run_id(idempotency_key)
    operation_id = str(uuid.uuid4())
    response_id = f"resp_{uuid.uuid4().hex}"
    now = datetime.now(UTC)
    queued = active_response(
        request,
        response_id=response_id,
        created_at=now.timestamp(),
    )
    initial_call_ids = _input_call_ids(messages)
    accepted = await background.create(
        NewRun(
            run_id=response_id,
            response_id=response_id,
            owner_scope=owner_scope,
            model=request.model,
            checkpoint_thread_id=checkpoint_key(
                request.model,
                operation_id,
                scope=f"background:{owner_scope}",
            ),
            graph_version=policy.version,
            envelope=cast(
                "dict[str, JsonValue]",
                request.model_dump(mode="json", by_alias=True),
            ),
            request_fingerprint=_background_fingerprint(request),
            idempotency_key=idempotency_key,
            response=cast("dict[str, JsonValue]", response_json(queued)),
            created_at=now,
            initial_call_ids=initial_call_ids,
            initial_message_count=len(messages),
        )
    )
    if accepted.response is None:
        msg = "Accepted background run has no public Response snapshot."
        raise RuntimeError(msg)
    return Response.model_validate(accepted.response)


async def retrieve_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Read one authorized snapshot without scheduling side effects."""
    if background is None:
        raise BackgroundResponseNotFoundError(response_id)
    owner_scope = normalize_checkpoint_scope(checkpoint_scope)
    run = await background.retrieve(response_id, owner_scope)
    if run is None or run.response is None:
        raise BackgroundResponseNotFoundError(response_id)
    return Response.model_validate(run.response)


async def cancel_background_response(
    response_id: str,
    background: BackgroundBackend | None,
    *,
    checkpoint_scope: str,
) -> Response:
    """Atomically choose logical cancellation or return the terminal winner."""
    if background is None:
        raise BackgroundResponseNotFoundError(response_id)
    owner_scope = normalize_checkpoint_scope(checkpoint_scope)
    current = await background.retrieve(response_id, owner_scope)
    if current is None or current.response is None:
        raise BackgroundResponseNotFoundError(response_id)
    if current.terminal and current.status is not ResponseStatus.CANCELLED:
        return Response.model_validate(current.response)

    request = ResponseCreateRequest.model_validate(current.envelope)
    cancellation_json = (
        current.response
        if current.status is ResponseStatus.CANCELLED
        else response_json(
            cancelled_response(
                request,
                response_id=current.response_id,
                created_at=current.created_at.timestamp(),
            )
        )
    )
    cancelled = await background.cancel(
        response_id,
        owner_scope,
        cast("dict[str, JsonValue]", cancellation_json),
    )
    if cancelled is None or cancelled.response is None:
        raise BackgroundResponseNotFoundError(response_id)
    return Response.model_validate(cancelled.response)


def _background_fingerprint(request: ResponseCreateRequest) -> str:
    normalized = request.model_copy(
        update={
            "background": True,
            "stream": False,
            "store": bool(request.store),
        }
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata = {
            key: value for key, value in metadata.items() if key != "lgos_run_id"
        }
        if metadata:
            normalized["metadata"] = metadata
        else:
            normalized.pop("metadata", None)
    canonical = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _input_call_ids(messages: list[BaseMessage]) -> tuple[str, ...]:
    call_ids: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            call_ids.extend(
                call_id
                for call in (*message.tool_calls, *message.invalid_tool_calls)
                if isinstance((call_id := call.get("id")), str)
            )
        elif isinstance(message, ToolMessage):
            call_ids.append(message.tool_call_id)
    return tuple(call_ids)


async def prepare_response_run(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    *,
    checkpoint_scope: str,
) -> GraphRun:
    """Validate a Responses request and prepare its graph run."""
    graph_config = graph_registry.get_graph(request.model)
    validate_tools(request, graph_config.server_tools)
    if request.previous_response_id is not None and not graph_config.supports(
        GraphFeature.INTERRUPTS
    ):
        message = (
            "Previous response state is not supported for model "
            f"'{request.model}'; only interruptible graphs support "
            "'previous_response_id'."
        )
        raise UnsupportedResponsesRequestError(
            message,
            param="previous_response_id",
        )
    graph_request, messages, resume = decode_responses_request(
        request,
        graph_config.server_tools,
    )
    return await prepare_run(
        graph_request,
        messages,
        graph_registry,
        resume=resume,
        checkpoint_scope=checkpoint_scope,
    )


async def collect_response(request: ResponseCreateRequest, run: GraphRun) -> Response:
    """Build one non-streaming Response from the graph's durable output."""
    try:
        server_tools = selected_server_tools(request, run.config.server_tools)
        builder = ResponsesEventBuilder(
            request,
            run_id=run.run_id,
            server_tools=server_tools,
        )
    except BaseException as exc:
        run.record_failure(exc)
        await run.aclose()
        raise

    if not server_tools:
        async with run:
            output = await invoke_run(run)
            for event in _terminal_events(builder, output, run):
                if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                    return event.response
    else:
        events = _successful_response_events(
            builder,
            run,
            stream_updates=True,
            streaming=False,
        )
        async with aclosing(events):
            async for event in events:
                if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                    return event.response
    msg = "Graph execution completed without a final Response."
    raise UnsupportedResponsesOutputError(msg)


async def stream_response(
    request: ResponseCreateRequest,
    run: GraphRun,
) -> AsyncGenerator[str, None]:
    """
    Stream one prepared graph run as a typed Responses lifecycle.

    Yields:
        Named, compact Responses SSE frames.

    """
    server_tools = selected_server_tools(request, run.config.server_tools)
    builder = ResponsesEventBuilder(
        request,
        run_id=run.run_id,
        server_tools=server_tools,
    )
    events = _successful_response_events(
        builder,
        run,
        stream_updates=bool(server_tools),
    )
    try:
        async with aclosing(events):
            async for event in events:
                yield encode_event(event)
    except Exception:
        logger.exception("responses.stream_failed")
        for response_event in builder.failure("Internal server error"):
            yield encode_event(response_event)


async def _successful_response_events(
    builder: ResponsesEventBuilder,
    run: GraphRun,
    *,
    stream_updates: bool,
    streaming: bool = True,
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """
    Adapt one successful graph stream to typed Responses events.

    Yields:
        The successful Response lifecycle.

    """
    final_output: AIMessage | LangGraphInterruptBatch | None = None
    async with run:
        yield builder.created()
        yield builder.in_progress()

        expose_status = streaming and run.config.supports(GraphFeature.CLIENT_EVENTS)
        run_events = stream_run(
            run,
            stream_messages=streaming,
            stream_updates=stream_updates,
        )
        async with aclosing(run_events):
            async for graph_event in run_events:
                if isinstance(graph_event, (AIMessage, LangGraphInterruptBatch)):
                    final_output = graph_event
                    continue
                for event in _graph_response_events(
                    builder,
                    graph_event,
                    expose_status=expose_status,
                ):
                    yield event

    for event in _terminal_events(builder, final_output, run):
        yield event


def _terminal_events(
    builder: ResponsesEventBuilder,
    output: AIMessage | LangGraphInterruptBatch | None,
    run: GraphRun,
) -> Iterator[ResponseStreamEvent]:
    if isinstance(output, LangGraphInterruptBatch):
        yield from builder.finish_interrupt(
            output,
            usage=response_usage(run.usage_metadata()),
        )
        return
    if output is None:
        msg = "LangGraph stream completed without a final assistant message."
        raise RuntimeError(msg)
    yield from builder.finish(output)


def _graph_response_events(
    builder: ResponsesEventBuilder,
    event: str | CustomStreamPart | UpdatesStreamPart,
    *,
    expose_status: bool,
) -> Iterator[ResponseStreamEvent]:
    """
    Translate one non-final graph event.

    Yields:
        Zero or more typed Responses events.

    """
    if isinstance(event, str):
        yield from builder.final_delta(event)
        return
    if event["type"] == "updates":
        yield from builder.server_tools(event)
        return
    if not expose_status:
        return

    status_data = parse_status_event(event["data"])
    if status_data is None or status_data.hidden:
        return
    yield from builder.commentary(status_data.description)


__all__ = [
    "BackgroundResponseNotFoundError",
    "accept_background_response",
    "cancel_background_response",
    "collect_response",
    "prepare_response_run",
    "retrieve_background_response",
    "stream_response",
]
