"""Responses API helpers for Open WebUI models."""

import asyncio
import json as responses_json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Union

import openai.types.responses as response_types
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from openai.types.responses import (
    CustomToolParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ToolParam,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import TypeAdapter

from .api import _model_request
from .contracts import (
    DISPLAY_FILE_TOOL_NAME,
    PACKAGE_VERSION_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    DisplayFileArguments,
    OpenWebUIEventEmitter,
    OpenWebUIMCPTool,
    OpenWebUIMessage,
    OpenWebUIMetadata,
    is_server_tool_model,
    supports_display_file,
    supports_web_search,
)
from .gateway import MCP_GATEWAY_ID


def _patch_legacy_custom_tool_output() -> None:
    """Fill the response-output omission in OpenAI 2.29 used by Open WebUI."""
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

    compatible_item = Union[ResponseOutputItem, ResponseCustomToolCallOutput]
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
    metadata: OpenWebUIMetadata,
) -> list[ToolParam]:
    """Build the tools owned by the selected demo client and graph."""
    tools: list[ToolParam] = (
        [DISPLAY_FILE_TOOL] if supports_display_file(model_id) else []
    )
    if (
        is_server_tool_model(model_id)
        and metadata.chat_variables.get(PACKAGE_VERSION_TOOL_NAME) is True
    ):
        tools.append(PACKAGE_VERSION_TOOL)
    if (
        supports_web_search(model_id)
        and metadata.chat_variables.get(WEB_SEARCH_TOOL_NAME) is True
    ):
        tools.append({"type": "web_search"})
    return tools


def _openwebui_mcp_tools(
    tools: Mapping[str, OpenWebUIMCPTool],
) -> tuple[list[FunctionToolParam], dict[str, str]]:
    """Translate managed Open WebUI tools to their gateway names."""
    translated = []
    openwebui_names = {}
    name_prefix = f"{MCP_GATEWAY_ID}_"
    for name, tool in tools.items():
        if not name.startswith(name_prefix):
            continue
        gateway_name = name.removeprefix(name_prefix)
        if not gateway_name:
            continue
        parameters: dict[str, object] = (
            dict(tool.spec.parameters)
            if tool.spec.parameters is not None
            else {"type": "object", "properties": {}}
        )
        translated_tool: FunctionToolParam = {
            "type": "function",
            "name": gateway_name,
            "parameters": parameters,
            "strict": tool.spec.strict,
        }
        if tool.spec.description is not None:
            translated_tool["description"] = tool.spec.description
        translated.append(translated_tool)
        openwebui_names[gateway_name] = name
    return translated, openwebui_names


def _openwebui_text_chunk(model_id: str, content: str) -> dict[str, Any]:
    """Keep text inside JSON: the Pipe host treats raw data: strings as SSE."""
    return ChatCompletionChunk(
        id="chatcmpl-lgos-responses",
        object="chat.completion.chunk",
        created=0,
        model=model_id,
        choices=[
            Choice(index=0, delta=ChoiceDelta(content=content), finish_reason=None)
        ],
    ).model_dump(exclude_none=True)


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
        if isinstance(content, str) and content:
            items.append({**message_fields, "content": content})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "input_text"} and isinstance(
                    part.get("text"), str
                ):
                    parts.append({"type": "input_text", "text": part["text"]})
                elif part.get("type") == "input_file":
                    file_id = part.get("file_id")
                    if isinstance(file_id, str) and file_id:
                        parts.append({"type": "input_file", "file_id": file_id})
            if parts:
                items.append({**message_fields, "content": parts})

        if role != "assistant" or not mcp_tool_names:
            continue
        for tool_call in message.tool_calls:
            if not tool_call.id or tool_call.function is None:
                continue
            gateway_name = (
                mcp_tool_names.get(tool_call.function.name)
                if tool_call.function.name is not None
                else None
            )
            if gateway_name is None:
                continue
            mcp_call_ids.add(tool_call.id)
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": gateway_name,
                    "arguments": tool_call.function.arguments or "{}",
                    "status": "completed",
                }
            )
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
    return {
        "id": "chatcmpl-lgos-responses",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        _openwebui_tool_call(
                            call,
                            name=openwebui_names[call.name],
                            index=index,
                        )
                        for index, call in enumerate(calls)
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _openwebui_tool_call(
    call: ResponseFunctionToolCall,
    *,
    name: str,
    index: int | None = None,
) -> dict[str, Any]:
    result = {
        "id": call.call_id,
        "type": "function",
        "function": {"name": name, "arguments": call.arguments},
    }
    if index is not None:
        result["index"] = index
    return result


def _responses_request(
    model_id: str,
    input_items: list[dict[str, Any]],
    metadata: dict[str, str] | None,
    user_id: str | None,
    *,
    background: bool,
    provider_routing: bool,
    tools: list[ToolParam],
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    request = {
        **_model_request(
            model_id,
            provider_routing=provider_routing,
        ),
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
    extra_headers = request.get("extra_headers")
    background_request = dict(request)
    idempotency_key = str(uuid.uuid4())
    if provider_routing:
        background_request["extra_headers"] = {
            **(extra_headers if isinstance(extra_headers, Mapping) else {}),
            "Idempotency-Key": idempotency_key,
        }
    else:
        background_request["extra_body"] = {
            "extra_headers": {"Idempotency-Key": idempotency_key}
        }
    response = await client.responses.create(**background_request)
    previous_status = None
    try:
        while response.status in ACTIVE_BACKGROUND_STATUSES:
            if response.status != previous_status:
                await on_status(response.status)
                previous_status = response.status
            await asyncio.sleep(1)
            response = await client.responses.retrieve(
                response.id,
                extra_headers=extra_headers,
            )
    except asyncio.CancelledError:
        try:
            await asyncio.shield(
                client.responses.cancel(
                    response.id,
                    extra_headers=extra_headers,
                )
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
        raise RuntimeError(
            f"Response incomplete: {reason.reason if reason else 'unknown reason'}."
        )
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
