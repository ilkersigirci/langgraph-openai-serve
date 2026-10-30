"""Convert graph output into OpenAI Response models."""

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, UsageMetadata
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
from langgraph_openai_serve.api.tools import encode_function_calls
from langgraph_openai_serve.graph.citations import citations_from_message
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.protocol import INTERRUPT_TOOL_NAME


@dataclass(frozen=True)
class ResponseContext:
    """Stable identity and request fields shared by one response lifecycle."""

    request: ResponseCreateRequest
    id: str
    created_at: int

    @classmethod
    def for_run(
        cls, request: ResponseCreateRequest, run_id: str | None
    ) -> "ResponseContext":
        """Identify a new Response, binding it to its interrupt run, if any."""
        return cls(
            request=request,
            id=(
                interrupt_response_id(run_id)
                if run_id is not None
                else f"resp_{uuid.uuid4().hex}"
            ),
            # OpenAI sends whole Unix seconds; strict clients reject fractions.
            created_at=int(time.time()),
        )

    def response(
        self,
        *,
        status: Literal["queued", "in_progress", "completed", "failed", "incomplete"],
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
                "background": bool(request.background),
                "completed_at": int(time.time()) if status == "completed" else None,
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
                "safety_identifier": request.safety_identifier,
                "service_tier": "default",
                "store": bool(request.store),
                "text": {"format": {"type": "text"}},
                "tool_choice": (
                    request.tool_choice.model_dump(mode="json")
                    if request.tool_choice is not None
                    and not isinstance(request.tool_choice, str)
                    else request.tool_choice or "auto"
                ),
                "tools": [
                    tool.model_dump(mode="json", by_alias=True)
                    for tool in request.tools or ()
                ],
                "top_logprobs": 0,
                "truncation": "disabled",
                "usage": usage,
                "user": request.user,
            }
        )


def response_function_calls(message: AIMessage) -> list[ResponseFunctionToolCall]:
    """Serialize the client tool calls of the final assistant message."""
    # A truncated message keeps its incomplete calls, as OpenAI returns them.
    return [
        _function_call_item(
            call_id=call.call_id, name=call.name, arguments=call.arguments
        )
        for call in encode_function_calls(
            message,
            include_invalid=response_incomplete_details(message) is not None,
        )
    ]


def interrupt_output_items(
    batch: LangGraphInterruptBatch,
) -> list[ResponseFunctionToolCall]:
    """Serialize one durable interrupt batch as function-call items."""
    return [
        _function_call_item(
            call_id=interrupt_tool_call_id(interrupt.id),
            name=INTERRUPT_TOOL_NAME,
            arguments=json.dumps(interrupt.value, separators=(",", ":")),
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
    # langchain-openai's Chat Completions path keeps refusals only here.
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
