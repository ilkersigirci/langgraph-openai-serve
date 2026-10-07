"""Responses API helpers for Open WebUI models."""

import asyncio
import json as responses_json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import openai.types.responses as response_types
from openai.types.responses import (
    CustomToolParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ToolParam,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import JsonValue, TypeAdapter

from .contracts import (
    DISPLAY_FILE_TOOL_NAME,
    PACKAGE_VERSION_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    DisplayFileArguments,
    OpenWebUIEventEmitter,
    OpenWebUIMessage,
    OpenWebUIMessageToolCall,
    OpenWebUIToolSpec,
    supports_display_file,
)
from .gateway import MCP_GATEWAY_ID


def _patch_legacy_custom_tool_output() -> None:
    """Fill the response-output omission in the OpenAI SDK shipped by Open WebUI."""
    if hasattr(response_types, "ResponseCustomToolCallOutputItem"):
        return

    from openai.types.responses.response_custom_tool_call_output import (
        ResponseCustomToolCallOutput,
    )
    from openai.types.responses.response_output_item_added_event import (
        ResponseOutputItemAddedEvent,
    )
    from openai.types.responses.response_output_item_done_event import (
        ResponseOutputItemDoneEvent,
    )

    compatible_item = ResponseOutputItem | ResponseCustomToolCallOutput
    fields = (
        (Response, "output", list[compatible_item]),
        (ResponseOutputItemAddedEvent, "item", compatible_item),
        (ResponseOutputItemDoneEvent, "item", compatible_item),
    )
    for model, field_name, annotation in fields:
        model.model_fields[field_name].annotation = annotation
        model.model_rebuild(force=True)


_patch_legacy_custom_tool_output()
RESPONSE_OUTPUT_ITEM = TypeAdapter(ResponseOutputItem)

DISPLAY_FILE_TOOL: FunctionToolParam = {
    "type": "function",
    "name": DISPLAY_FILE_TOOL_NAME,
    "description": "Display a file stored in the configured OpenAI Files API.",
    "strict": True,
    "parameters": DisplayFileArguments.model_json_schema(),
}
PACKAGE_VERSION_TOOL: CustomToolParam = {
    "type": "custom",
    "name": PACKAGE_VERSION_TOOL_NAME,
}
ACTIVE_BACKGROUND_STATUSES = {"queued", "in_progress"}
logger = logging.getLogger(__name__)


def _responses_tools(
    model_id: str,
    chat_variables: Mapping[str, JsonValue],
) -> list[ToolParam]:
    """Build the tools owned by the selected demo client and graph."""
    tools: list[ToolParam] = (
        [DISPLAY_FILE_TOOL] if supports_display_file(model_id) else []
    )
    # Only the generated server-tool and advanced-graph models declare these.
    if chat_variables.get(PACKAGE_VERSION_TOOL_NAME) is True:
        tools.append(PACKAGE_VERSION_TOOL)
    if chat_variables.get(WEB_SEARCH_TOOL_NAME) is True:
        tools.append({"type": "web_search"})
    return tools


def _openwebui_mcp_tools(
    tools: Mapping[str, Mapping[str, Any]],
) -> tuple[list[FunctionToolParam], dict[str, str]]:
    """Translate gateway MCP tools from Open WebUI's ``__tools__`` map."""
    translated = []
    openwebui_names = {}
    for name, tool in tools.items():
        # Open WebUI prefixes each MCP tool with its connection ID.
        gateway_name = name.removeprefix(f"{MCP_GATEWAY_ID}_")
        if gateway_name == name or not gateway_name:
            continue
        spec = OpenWebUIToolSpec.model_validate(tool["spec"])
        parameters: dict[str, object] = (
            dict(spec.parameters)
            if spec.parameters is not None
            else {"type": "object", "properties": {}}
        )
        translated_tool: FunctionToolParam = {
            "type": "function",
            "name": gateway_name,
            "parameters": parameters,
            "strict": spec.strict,
        }
        if spec.description is not None:
            translated_tool["description"] = spec.description
        translated.append(translated_tool)
        openwebui_names[gateway_name] = name
    return translated, openwebui_names


def _openwebui_chunk(
    model_id: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Return a stream chunk; the Pipe host treats raw ``data:`` strings as SSE."""
    return {
        "id": "chatcmpl-lgos-responses",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _responses_content(content: JsonValue) -> str | list[dict[str, str]]:
    """Retain supported text and file parts from one transcript message."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return []
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if part.get("type") in {"text", "input_text"} and isinstance(text, str):
            parts.append({"type": "input_text", "text": text})
        elif part.get("type") == "input_file":
            file_id = part.get("file_id")
            if isinstance(file_id, str) and file_id:
                parts.append({"type": "input_file", "file_id": file_id})
    return parts


def _responses_mcp_calls(
    tool_calls: Sequence[OpenWebUIMessageToolCall], tool_names: Mapping[str, str]
) -> list[dict[str, str]]:
    """Translate only calls owned by the configured MCP gateway."""
    calls = []
    for tool_call in tool_calls:
        if (
            not tool_call.id
            or tool_call.function is None
            or tool_call.function.name is None
        ):
            continue
        gateway_name = tool_names.get(tool_call.function.name)
        if gateway_name is None:
            continue
        calls.append(
            {
                "type": "function_call",
                "call_id": tool_call.id,
                "name": gateway_name,
                "arguments": tool_call.function.arguments or "{}",
                "status": "completed",
            }
        )
    return calls


def _responses_input(
    messages: Sequence[OpenWebUIMessage],
    *,
    mcp_tool_names: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert Open WebUI's text/file transcript into Responses items."""
    items = []
    mcp_call_ids: set[str] = set()
    for message in messages:
        role = message.role
        content = message.content
        if role == "tool":
            if message.tool_call_id in mcp_call_ids:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": _tool_output(content),
                    }
                )
            continue
        if role not in {"user", "assistant", "system", "developer"}:
            continue
        message_fields = {"role": role}
        if role == "assistant":
            message_fields["phase"] = message.phase or "final_answer"
        if response_content := _responses_content(content):
            items.append({**message_fields, "content": response_content})

        if role == "assistant" and mcp_tool_names:
            calls = _responses_mcp_calls(message.tool_calls, mcp_tool_names)
            mcp_call_ids.update(call["call_id"] for call in calls)
            items.extend(calls)
    return items


def _tool_output(content: object) -> str:
    if isinstance(content, str):
        return content
    return responses_json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _openwebui_tool_chunk(
    model_id: str,
    calls: list[ResponseFunctionToolCall],
    openwebui_names: Mapping[str, str],
) -> dict[str, Any]:
    """Return graph calls for execution by Open WebUI's native tool loop."""
    tool_calls = [
        {
            "index": index,
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": openwebui_names[call.name],
                "arguments": call.arguments,
            },
        }
        for index, call in enumerate(calls)
    ]
    return _openwebui_chunk(
        model_id, {"tool_calls": tool_calls}, finish_reason="tool_calls"
    )


def _responses_request(
    model_id: str,
    input_items: list[dict[str, Any]],
    metadata: dict[str, str] | None,
    user_id: str | None,
    *,
    background: bool,
    tools: list[ToolParam],
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    request = {
        "model": model_id,
        "input": input_items,
        "store": background,
        "tools": tools,
    }
    if background:
        request["background"] = True
    if metadata:
        request["metadata"] = metadata
    if user_id is not None:
        request["user"] = user_id
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    return request


async def _background_response(
    client: Any,
    request: dict[str, Any],
    on_status: Callable[[str], Awaitable[None]],
    *,
    provider_routing: bool,
) -> Response:
    """Create and poll one background Response with best-effort cancellation."""
    client = client.with_options(max_retries=2)
    background_request = dict(request)
    idempotency_key = str(uuid.uuid4())
    lifecycle_options: dict[str, Any] = {}
    if provider_routing:
        background_request["extra_headers"] = {"Idempotency-Key": idempotency_key}
        # Retrieve and cancel carry no model; without this query parameter
        # Bifrost routes them to its built-in openai provider.
        provider = request["model"].partition("/")[0]
        lifecycle_options["extra_query"] = {"provider": provider}
    else:
        background_request["extra_body"] = {
            "extra_headers": {"Idempotency-Key": idempotency_key}
        }
    response = await client.responses.create(**background_request)
    previous_status = None
    try:  # ruff: ignore[too-many-statements-in-try-clause] - Cancellation must cancel the remote response throughout polling.
        while response.status in ACTIVE_BACKGROUND_STATUSES:
            if response.status != previous_status:
                await on_status(response.status)
                previous_status = response.status
            await asyncio.sleep(1)
            response = await client.responses.retrieve(response.id, **lifecycle_options)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(
                client.responses.cancel(response.id, **lifecycle_options)
            )
        except Exception:
            logger.warning(
                "Background response cancellation failed for %s",
                response.id,
                exc_info=True,
            )
        raise
    return response


def _responses_final_text(response: Response) -> str:
    """Select only durable final-answer messages."""
    parts = []
    for item in response.output:
        if item.type != "message" or item.phase == "commentary":
            continue
        parts.extend(
            part.text if part.type == "output_text" else part.refusal
            for part in item.content
        )
    return "".join(parts)


def _raise_for_response(response: Response) -> None:
    if response.status == "completed":
        return
    if response.status == "incomplete":
        reason = response.incomplete_details
        msg = f"Response incomplete: {reason.reason if reason else 'unknown reason'}."
        raise RuntimeError(msg)
    detail = response.error
    raise RuntimeError(detail.message if detail is not None else "Response failed.")


def _responses_function_calls(
    response: Response,
) -> list[ResponseFunctionToolCall]:
    """Return client-owned function calls from a completed Response."""
    return [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]


def _responses_continuation(
    response: Response,
    outputs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        *(
            item.model_dump(mode="json", exclude_none=True)
            if item.type == "custom_tool_call_output"
            else RESPONSE_OUTPUT_ITEM.dump_python(
                item,
                mode="json",
                exclude_none=True,
            )
            for item in response.output
        ),
        *outputs,
    ]


async def _emit_response_sources(
    response: Response, event_emitter: OpenWebUIEventEmitter | None
) -> None:
    """Use complete, typed annotations instead of accumulating citation deltas."""
    if event_emitter is None:
        return
    for item in response.output:
        if item.type != "message" or item.phase == "commentary":
            continue
        for part in item.content:
            if part.type != "output_text":
                continue
            for annotation in part.annotations:
                if annotation.type == "url_citation":
                    event = _openwebui_source_event(annotation, part.text)
                    if event is not None:
                        await event_emitter(event)


def _openwebui_source_event(
    annotation: AnnotationURLCitation,
    text: str,
) -> dict[str, Any] | None:
    """Translate one Responses URL annotation into a persistent UI source."""
    stop = annotation.end_index + 1
    if not 0 <= annotation.start_index < stop <= len(text):
        return None

    cited_text = text[annotation.start_index : stop]
    return {
        "type": "source",
        "data": {
            "source": {"name": annotation.title, "url": annotation.url},
            "document": [cited_text],
            "metadata": [
                {
                    "source": annotation.title,
                    "name": annotation.title,
                    "url": annotation.url,
                }
            ],
        },
    }
