"""Build and commit canonical OpenAI Response snapshots for background runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import add_usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseError,
    ResponseIncompleteEvent,
)

from langgraph_openai_serve.api.responses.events import ResponsesEventBuilder
from langgraph_openai_serve.api.responses.output import (
    ResponseContext,
    UnsupportedResponsesOutputError,
    response_usage,
)
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.protocol import INTERRUPT_TOOL_NAME

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from langchain_core.messages.ai import UsageMetadata
    from pydantic import JsonValue

    from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
    from langgraph_openai_serve.background.contracts import BackgroundSettings
    from langgraph_openai_serve.background.store import ResponseStore, StoredRun


def queued_response(
    request: ResponseCreateRequest,
    *,
    response_id: str,
    created_at: float,
) -> Response:
    """Build the queued snapshot of a newly accepted Response."""
    return ResponseContext.for_run(
        request,
        response_id=response_id,
        created_at=created_at,
    ).response(status="queued", output=[])


def cancelled_response(response: Response) -> Response:
    """Turn a persisted Response snapshot into a cancellation winner."""
    return response.model_copy(
        update={
            "status": "cancelled",
            "output": [],
            "error": None,
            "incomplete_details": None,
        }
    )


def failed_response(
    response: Response,
    *,
    message: str,
) -> Response:
    """Turn a persisted Response snapshot into a terminal failure."""
    return response.model_copy(
        update={
            "status": "failed",
            "output": [],
            "error": ResponseError.model_validate(
                {
                    # ResponseError.code is an OpenAI-owned closed vocabulary. Keep
                    # LGOS's more specific operational reason in structured logs
                    # rather than inventing a nonstandard wire value.
                    "code": "server_error",
                    "message": message,
                    "misalignment": None,
                }
            ),
            "incomplete_details": None,
        }
    )


def output_response(  # ruff: ignore[too-many-arguments] - Rendering needs the persisted Response identity and transcript boundary.
    request: ResponseCreateRequest,
    output: AIMessage | LangGraphInterruptBatch,
    *,
    response_id: str,
    created_at: float,
    attempt_usage: UsageMetadata | None = None,
    server_tools: Sequence[str] = (),
    root_messages: Iterable[BaseMessage] = (),
    prior_ids: frozenset[str] = frozenset(),
) -> Response:
    """
    Render checkpointed output through the existing Responses builder.

    Transcript items named by ``prior_ids`` precede this Response, so they are
    neither rendered nor counted again. Pending interrupts become the same
    ``lgos_interrupt`` function calls a foreground run returns.
    """
    messages = tuple(
        message for message in root_messages if message.id not in prior_ids
    )
    builder = ResponsesEventBuilder(
        request,
        response_id=response_id,
        created_at=created_at,
        server_tools=server_tools,
    )
    if server_tools:
        operation_messages = (
            visible
            for message in messages
            if (visible := _new_operation_message(message, prior_ids)) is not None
        )
        tuple(builder.server_tool_messages(operation_messages))

    # Checkpointed messages keep usage across retried deliveries, which the
    # per-attempt callbacks cannot.
    usage = None
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            usage = add_usage(usage, message.usage_metadata)
    usage = usage or attempt_usage
    events = (
        builder.finish_interrupt(output, usage=response_usage(usage))
        if isinstance(output, LangGraphInterruptBatch)
        else builder.finish(
            output.model_copy(update={"usage_metadata": usage}) if usage else output
        )
    )
    for event in events:
        if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
            return event.response
    msg = "Background output rendering produced no terminal Response."
    raise UnsupportedResponsesOutputError(msg)


def _new_operation_message(
    message: BaseMessage,
    prior_ids: frozenset[str],
) -> BaseMessage | None:
    """Drop tool activity that precedes this Response."""
    if isinstance(message, ToolMessage):
        return None if message.tool_call_id in prior_ids else message
    if not isinstance(message, AIMessage):
        return None
    calls = [call for call in message.tool_calls if call.get("id") not in prior_ids]
    invalid_calls = [
        call for call in message.invalid_tool_calls if call.get("id") not in prior_ids
    ]
    if not calls and not invalid_calls:
        return None
    return message.model_copy(
        update={"tool_calls": calls, "invalid_tool_calls": invalid_calls}
    )


def transcript_ids(messages: Iterable[BaseMessage]) -> tuple[str, ...]:
    """Return the message and tool-call IDs that identify transcript items."""
    ids: list[str] = []
    for message in messages:
        if message.id:
            ids.append(message.id)
        if isinstance(message, AIMessage):
            ids.extend(
                call_id
                for call in (*message.tool_calls, *message.invalid_tool_calls)
                if isinstance((call_id := call.get("id")), str)
            )
        elif isinstance(message, ToolMessage):
            ids.append(message.tool_call_id)
    return tuple(ids)


def is_paused(response: dict[str, JsonValue]) -> bool:
    """Whether a stored Response waits for interrupt answers."""
    output = response.get("output")
    return isinstance(output, list) and any(
        isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") == INTERRUPT_TOOL_NAME
        for item in output
    )


def response_json(response: Response) -> dict[str, JsonValue]:
    """Serialize one typed Response for a JSON storage boundary."""
    return cast(
        "dict[str, JsonValue]",
        response.model_dump(mode="json", by_alias=True),
    )


async def finish_response(
    store: ResponseStore,
    settings: BackgroundSettings,
    response_id: str,
    response: Response,
) -> StoredRun | None:
    """Commit a terminal Response with its retention and return the winner."""
    return await store.finish(
        response_id,
        response_json(response),
        now=datetime.now(UTC),
        result_retention=settings.result_retention_for(response),
    )


__all__ = [
    "cancelled_response",
    "failed_response",
    "finish_response",
    "is_paused",
    "output_response",
    "queued_response",
    "response_json",
    "transcript_ids",
]
