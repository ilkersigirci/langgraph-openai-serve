"""Build canonical OpenAI Response snapshots for background execution."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import TYPE_CHECKING, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from langgraph.types import UpdatesStreamPart
    from pydantic import JsonValue

    from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest


def active_response(
    request: ResponseCreateRequest,
    *,
    response_id: str,
    created_at: float,
    status: Literal["queued", "in_progress"] = "queued",
) -> Response:
    """Build a queued or in-progress snapshot with no speculative output."""
    if status not in {"queued", "in_progress"}:
        msg = "An active background Response must be queued or in_progress."
        raise ValueError(msg)
    return ResponseContext.for_run(
        request,
        response_id=response_id,
        created_at=created_at,
    ).response(status=status, output=[])


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


def is_stored_response(response: Response) -> bool:
    """Return the retention choice preserved on the Response snapshot."""
    return response.model_dump(mode="json").get("store") is True


def output_response(  # ruff: ignore[too-many-arguments] - Rendering needs the persisted Response identity and transcript boundary.
    request: ResponseCreateRequest,
    message: AIMessage,
    *,
    response_id: str,
    created_at: float,
    server_tools: Sequence[str] = (),
    root_messages: Iterable[BaseMessage] = (),
    initial_call_ids: frozenset[str] = frozenset(),
) -> Response:
    """Render checkpointed output through the existing Responses builder."""
    builder = ResponsesEventBuilder(
        request,
        response_id=response_id,
        created_at=created_at,
        server_tools=server_tools,
    )
    if server_tools:
        for root_message in root_messages:
            visible = _new_operation_message(root_message, initial_call_ids)
            if visible is None:
                continue
            update = cast(
                "UpdatesStreamPart",
                {
                    "type": "updates",
                    "ns": (),
                    "data": {"background_recovery": {"messages": [visible]}},
                },
            )
            tuple(builder.server_tools(update))

    terminal: Response | None = None
    for event in builder.finish(message):
        if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
            terminal = event.response
    if terminal is None:
        msg = "Background output rendering produced no terminal Response."
        raise UnsupportedResponsesOutputError(msg)
    return _stable_output_ids(terminal)


def _new_operation_message(
    message: BaseMessage,
    initial_call_ids: frozenset[str],
) -> BaseMessage | None:
    if isinstance(message, ToolMessage):
        return None if message.tool_call_id in initial_call_ids else message
    if not isinstance(message, AIMessage):
        return None
    calls = [
        call for call in message.tool_calls if call.get("id") not in initial_call_ids
    ]
    invalid_calls = [
        call
        for call in message.invalid_tool_calls
        if call.get("id") not in initial_call_ids
    ]
    if not calls and not invalid_calls:
        return None
    return message.model_copy(
        update={"tool_calls": calls, "invalid_tool_calls": invalid_calls}
    )


def _stable_output_ids(response: Response) -> Response:
    payload = response.model_dump(mode="json", by_alias=True)
    output = payload.get("output")
    if not isinstance(output, list):
        return response
    for index, item in enumerate(output):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        item_type = str(item.get("type", "item"))
        prefix = {
            "message": "msg",
            "function_call": "fc",
            "custom_tool_call": "ctc",
            "custom_tool_call_output": "ctco",
            "web_search_call": "ws",
        }.get(item_type, "item")
        logical_identity = _logical_item_identity(item, index=index)
        identity = f"{response.id}:{item_type}:{logical_identity}".encode()
        item["id"] = f"{prefix}_{sha256(identity).hexdigest()[:24]}"
    return Response.model_validate(payload)


def _logical_item_identity(item: dict[str, object], *, index: int) -> str:
    call_id = item.get("call_id")
    if isinstance(call_id, str) and call_id:
        return f"call:{call_id}"
    if item.get("type") == "message":
        phase = item.get("phase")
        if isinstance(phase, str) and phase:
            return f"message:{phase}"
    item_id = item.get("id")
    if item.get("type") == "web_search_call" and isinstance(item_id, str):
        # Server-tool web search IDs are derived from the durable graph call ID.
        return f"web-search:{item_id}"
    canonical = {key: value for key, value in item.items() if key != "id"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"item:{index}:{sha256(encoded.encode()).hexdigest()}"


def response_json(response: Response) -> dict[str, JsonValue]:
    """Serialize one typed Response for a JSON storage boundary."""
    return cast(
        "dict[str, JsonValue]",
        response.model_dump(mode="json", by_alias=True),
    )


__all__ = [
    "active_response",
    "cancelled_response",
    "failed_response",
    "is_stored_response",
    "output_response",
    "response_json",
]
