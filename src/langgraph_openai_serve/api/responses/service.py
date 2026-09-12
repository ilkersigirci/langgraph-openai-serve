"""Execute and assemble OpenAI Response objects."""

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, InvalidToolCall, UsageMetadata
from langchain_core.messages.tool import ToolCall
from openai.types.responses import (
    Response,
    ResponseError,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_output_text import AnnotationURLCitation
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from langgraph_openai_serve.api.responses.interrupts import (
    interrupt_response_id,
    interrupt_tool_call_id,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.graph.citations import citations_from_message
from langgraph_openai_serve.graph.interrupt.models import LangGraphInterruptBatch
from langgraph_openai_serve.protocol import INTERRUPT_TOOL_NAME


class UnsupportedResponsesOutputError(RuntimeError):
    """Raised when graph output cannot be serialized as supported Responses items."""


@dataclass(frozen=True)
class ResponseContext:
    """Stable identity and request fields shared by one response lifecycle."""

    request: ResponseCreateRequest
    id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.time)

    @classmethod
    def for_run(
        cls,
        request: ResponseCreateRequest,
        *,
        run_id: str | None = None,
    ) -> "ResponseContext":
        """Build context, binding an interrupt response ID when run_id is present."""
        if run_id is None:
            return cls(request=request)
        return cls(request=request, id=interrupt_response_id(run_id))

    def response(
        self,
        *,
        status: Literal["in_progress", "completed", "failed", "incomplete"],
        output: Sequence[ResponseOutputItem],
        error: ResponseError | None = None,
        usage: ResponseUsage | None = None,
        incomplete_details: IncompleteDetails | None = None,
    ) -> Response:
        """Build one SDK-typed Response with the route's stable defaults."""
        request = self.request
        return Response.model_validate(
            {
                "id": self.id,
                "object": "response",
                "created_at": self.created_at,
                "status": status,
                "background": False,
                "completed_at": time.time() if status == "completed" else None,
                "error": error,
                "incomplete_details": incomplete_details,
                "instructions": request.instructions,
                "metadata": dict(request.metadata or {}),
                "model": request.model,
                "output": list(output),
                "parallel_tool_calls": (
                    request.parallel_tool_calls
                    if request.parallel_tool_calls is not None
                    else True
                ),
                "previous_response_id": request.previous_response_id,
                "service_tier": "default",
                "text": {"format": {"type": "text"}},
                "tool_choice": (
                    request.tool_choice.model_dump(mode="json")
                    if request.tool_choice is not None
                    and not isinstance(request.tool_choice, str)
                    else request.tool_choice or "auto"
                ),
                "tools": [tool.model_dump(mode="json") for tool in request.tools or ()],
                "top_logprobs": 0,
                "truncation": "disabled",
                "usage": usage,
                "user": request.user,
            }
        )


def response_function_calls(message: AIMessage) -> list[ResponseFunctionToolCall]:
    """Serialize and validate all client tool calls from an assistant message."""
    if message.invalid_tool_calls and response_incomplete_details(message) is None:
        msg = "The final assistant message contains invalid tool calls."
        raise UnsupportedResponsesOutputError(msg)

    calls = [response_function_call(call) for call in message.tool_calls]
    calls.extend(_incomplete_function_call(call) for call in message.invalid_tool_calls)
    seen_call_ids: set[str] = set()
    for output in calls:
        if output.call_id in seen_call_ids:
            msg = f"The final assistant message repeats call id '{output.call_id}'."
            raise UnsupportedResponsesOutputError(msg)
        seen_call_ids.add(output.call_id)
    return calls


def _incomplete_function_call(call: InvalidToolCall) -> ResponseFunctionToolCall:
    call_id, name, arguments = call.get("id"), call.get("name"), call.get("args")
    if not call_id or not name or arguments is None:
        msg = "The incomplete tool call must include an id, name, and arguments."
        raise UnsupportedResponsesOutputError(msg)
    return _function_call_item(call_id=call_id, name=name, arguments=arguments)


def response_function_call(call: ToolCall) -> ResponseFunctionToolCall:
    """Serialize one LangChain client tool call."""
    call_id = call.get("id")
    name = call.get("name")
    arguments = call.get("args")
    if not isinstance(call_id, str) or not call_id:
        msg = "The final assistant tool call must include a non-empty id."
        raise UnsupportedResponsesOutputError(msg)
    if not isinstance(name, str) or not name:
        msg = "The final assistant tool call must include a non-empty name."
        raise UnsupportedResponsesOutputError(msg)
    if not isinstance(arguments, dict):
        msg = "The final assistant tool call arguments must be a JSON object."
        raise UnsupportedResponsesOutputError(msg)
    return _function_call_item(
        call_id=call_id,
        name=name,
        arguments=_dump_arguments(arguments),
    )


def interrupt_output_items(
    batch: LangGraphInterruptBatch,
    *,
    response_id: str,
) -> list[ResponseFunctionToolCall]:
    """Serialize one durable interrupt batch as function-call items."""
    return [
        _function_call_item(
            call_id=interrupt_tool_call_id(
                interrupt.id,
                state_token=batch.state_token,
                response_id=response_id,
            ),
            name=INTERRUPT_TOOL_NAME,
            arguments=_dump_arguments(interrupt.value),
        )
        for interrupt in batch.interrupts
    ]


def _function_call_item(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{uuid.uuid4().hex}",
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
        type="function_call",
    )


def _dump_arguments(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        msg = "The final assistant tool call arguments must be valid JSON values."
        raise UnsupportedResponsesOutputError(msg) from exc


def response_output_text(message: AIMessage) -> ResponseOutputText:
    """Build final Responses text and validated native URL annotations."""
    return ResponseOutputText(
        annotations=[
            AnnotationURLCitation(
                type="url_citation",
                url=citation["url"],
                title=citation["title"],
                start_index=citation["start_index"],
                end_index=citation["end_index"],
            )
            for citation in citations_from_message(message)
        ],
        logprobs=[],
        text=str(message.text),
        type="output_text",
    )


def response_refusals(message: AIMessage) -> list[ResponseOutputRefusal]:
    """Read refusals through LangChain's normalized content boundary."""
    refusals = []
    for block in message.content_blocks:
        value = block.get("value")
        if block["type"] != "non_standard" or not isinstance(value, dict):
            continue
        refusal = value.get("refusal")
        if value.get("type") == "refusal" and isinstance(refusal, str):
            refusals.append(ResponseOutputRefusal(type="refusal", refusal=refusal))
    fallback = message.additional_kwargs.get("refusal")
    if not refusals and isinstance(fallback, str):
        refusals.append(ResponseOutputRefusal(type="refusal", refusal=fallback))
    return refusals


def response_incomplete_details(message: AIMessage) -> IncompleteDetails | None:
    """Keep the final provider's truncation or filtering outcome visible."""
    metadata = message.response_metadata
    if metadata.get("status") == "incomplete":
        return IncompleteDetails.model_validate(
            metadata.get("incomplete_details") or {}
        )
    reason = metadata.get("finish_reason")
    if reason == "length":
        return IncompleteDetails(reason="max_output_tokens")
    if reason == "content_filter":
        return IncompleteDetails(reason="content_filter")
    return None


def response_usage(usage: UsageMetadata | None) -> ResponseUsage | None:
    """Map provider-reported LangChain usage to Responses token details."""
    if usage is None:
        return None
    input_details = usage.get("input_token_details", {})
    output_details = usage.get("output_token_details", {})
    return ResponseUsage(
        input_tokens=usage["input_tokens"],
        input_tokens_details=InputTokensDetails(
            cached_tokens=input_details.get("cache_read", 0),
            cache_write_tokens=input_details.get("cache_creation", 0),
        ),
        output_tokens=usage["output_tokens"],
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=output_details.get("reasoning", 0),
        ),
        total_tokens=usage["total_tokens"],
    )


__all__ = [
    "ResponseContext",
    "UnsupportedResponsesOutputError",
    "interrupt_output_items",
    "response_function_call",
    "response_function_calls",
    "response_incomplete_details",
    "response_output_text",
    "response_refusals",
    "response_usage",
]
