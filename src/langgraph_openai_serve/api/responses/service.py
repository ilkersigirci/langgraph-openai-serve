"""Execute and assemble OpenAI Response objects."""

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, UsageMetadata
from langchain_core.messages.tool import ToolCall
from openai.types.responses import (
    Response,
    ResponseError,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.graph.citations import citations_from_message
from langgraph_openai_serve.graph.interrupt.codec import (
    INTERRUPT_TOOL_NAME,
    interrupt_arguments,
    interrupt_tool_call_id,
)
from langgraph_openai_serve.graph.interrupt.models import LangGraphInterruptBatch
from langgraph_openai_serve.graph.runner import invoke_run
from langgraph_openai_serve.graph.utils import GraphRun


class UnsupportedResponsesOutputError(RuntimeError):
    """Raised when graph output cannot be serialized as supported Responses items."""


@dataclass(frozen=True)
class ResponseContext:
    """Stable identity and request fields shared by one response lifecycle."""

    request: ResponseCreateRequest
    id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.time)


async def generate_response(
    request: ResponseCreateRequest,
    run: GraphRun,
) -> Response:
    """Invoke a graph and serialize its durable Responses output."""
    invocation = await invoke_run(run)
    output = invocation.output
    items: Sequence[ResponseOutputItem]
    if isinstance(output, AIMessage):
        items = response_output_items(output)
        usage = output.usage_metadata
    else:
        items = interrupt_output_items(output)
        usage = run.usage_metadata()
    return response_object(
        ResponseContext(request=request),
        status="completed",
        output=items,
        usage=response_usage(usage),
    )


def response_output_items(message: AIMessage) -> list[ResponseOutputItem]:
    """Serialize one assistant message into ordered Responses output items."""
    calls = response_function_calls(message)
    output: list[ResponseOutputItem] = []
    if message.text or not message.tool_calls:
        output.append(
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                content=[response_output_text(message)],
                role="assistant",
                status="completed",
                type="message",
                phase="final_answer",
            )
        )
    output.extend(calls)
    return output


def response_function_calls(message: AIMessage) -> list[ResponseFunctionToolCall]:
    """Serialize and validate all client tool calls from an assistant message."""
    if message.invalid_tool_calls:
        msg = "The final assistant message contains invalid tool calls."
        raise UnsupportedResponsesOutputError(msg)

    calls: list[ResponseFunctionToolCall] = []
    seen_call_ids: set[str] = set()
    for call in message.tool_calls:
        output = response_function_call(call)
        if output.call_id in seen_call_ids:
            msg = f"The final assistant message repeats call id '{output.call_id}'."
            raise UnsupportedResponsesOutputError(msg)
        seen_call_ids.add(output.call_id)
        calls.append(output)
    return calls


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
) -> list[ResponseFunctionToolCall]:
    """Serialize one durable interrupt batch as function-call items."""
    return [
        _function_call_item(
            call_id=interrupt_tool_call_id(interrupt.id),
            name=INTERRUPT_TOOL_NAME,
            arguments=interrupt_arguments(
                run_id=batch.run_id,
                state_token=batch.state_token,
                payload=interrupt.value,
            ),
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


def response_object(
    context: ResponseContext,
    *,
    status: Literal["in_progress", "completed", "failed"],
    output: Sequence[ResponseOutputItem],
    error: ResponseError | None = None,
    usage: ResponseUsage | None = None,
) -> Response:
    """Build one SDK-typed Response with the route's stable defaults."""
    request = context.request
    return Response.model_validate(
        {
            "id": context.id,
            "object": "response",
            "created_at": context.created_at,
            "status": status,
            "background": False,
            "completed_at": (
                context.created_at if status in {"completed", "failed"} else None
            ),
            "error": error,
            "incomplete_details": None,
            "instructions": request.instructions,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "metadata": dict(request.metadata or {}),
            "model": request.model,
            "output": list(output),
            "parallel_tool_calls": (
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            "previous_response_id": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": "default",
            "store": False,
            "temperature": None,
            "text": {"format": {"type": "text"}},
            "tool_choice": (
                request.tool_choice.model_dump(mode="json")
                if request.tool_choice is not None
                and not isinstance(request.tool_choice, str)
                else request.tool_choice or "auto"
            ),
            "tools": [tool.model_dump(mode="json") for tool in request.tools or ()],
            "top_logprobs": 0,
            "top_p": None,
            "truncation": "disabled",
            "usage": usage,
            "user": request.user,
        }
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


def response_usage(usage: UsageMetadata | None) -> ResponseUsage | None:
    """Map provider-reported LangChain usage to Responses token details."""
    if usage is None:
        return None
    return ResponseUsage(
        input_tokens=usage["input_tokens"],
        input_tokens_details=InputTokensDetails(
            cached_tokens=0,
            cache_write_tokens=0,
        ),
        output_tokens=usage["output_tokens"],
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=usage["total_tokens"],
    )


__all__ = [
    "ResponseContext",
    "UnsupportedResponsesOutputError",
    "generate_response",
    "interrupt_output_items",
    "response_function_call",
    "response_function_calls",
    "response_object",
    "response_output_items",
    "response_output_text",
    "response_usage",
]
