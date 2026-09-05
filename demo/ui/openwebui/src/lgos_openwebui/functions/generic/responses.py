"""Responses API helpers for Open WebUI models."""

from typing import Any

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import TypeAdapter, ValidationError

from .api import _model_request
from .contracts import DISPLAY_FILE_TOOL_NAME, DisplayFileArguments

RESPONSE_OUTPUT = TypeAdapter(list[ResponseOutputItem])

DISPLAY_FILE_TOOL = {
    "type": "function",
    "name": DISPLAY_FILE_TOOL_NAME,
    "description": "Display a file stored in the configured OpenAI Files API.",
    "strict": True,
    "parameters": DisplayFileArguments.model_json_schema(),
}


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Open WebUI's text/file transcript into Responses items."""
    items = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system", "developer"}:
            continue
        if isinstance(content, str):
            items.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

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
            elif part.get("type") == "file" and isinstance(part.get("file"), dict):
                file_id = part["file"].get("file_id")
                if isinstance(file_id, str) and file_id:
                    parts.append({"type": "input_file", "file_id": file_id})
        if parts:
            items.append({"role": role, "content": parts})
    return items


def _responses_request(
    model_id: str,
    input_items: list[dict[str, Any]],
    metadata: dict[str, str] | None,
    user_id: str | None,
    *,
    provider_routing: bool,
    model_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    request = {
        **_model_request(
            model_id,
            provider_routing=provider_routing,
            model_prefixes=model_prefixes,
        ),
        "input": input_items,
        "store": False,
        "tools": [DISPLAY_FILE_TOOL],
    }
    if metadata:
        request["metadata"] = metadata
    if user_id is not None:
        request["user"] = user_id
    return request


def _responses_final_text(response: Response) -> str:
    """Select only durable final-answer messages."""
    parts = []
    for item in response.output:
        if item.type != "message" or item.phase != "final_answer":
            continue
        parts.extend(part.text for part in item.content if part.type == "output_text")
    return "".join(parts)


def _responses_function_calls(
    response: Response,
) -> list[ResponseFunctionToolCall]:
    return [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]


def _responses_continuation(
    response: Response,
    outputs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        # Serialize wire types, excluding SDK-only parsed fields on subclasses.
        *RESPONSE_OUTPUT.dump_python(response.output, mode="json", exclude_none=True),
        *outputs,
    ]


def _responses_url_annotation(value: object) -> AnnotationURLCitation | None:
    """Validate the SDK event's deliberately untyped annotation payload."""
    try:
        return AnnotationURLCitation.model_validate(value)
    except ValidationError:
        return None


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


def _openwebui_text_chunk(model_id: str, content: str) -> dict[str, Any]:
    """Encode text for Open WebUI's Pipe interface, after Responses inference."""
    return {
        "id": "chatcmpl-lgos-responses",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


def _openwebui_finish_chunk(model_id: str) -> dict[str, Any]:
    value = _openwebui_text_chunk(model_id, "")
    value["choices"][0]["finish_reason"] = "stop"
    return value


def _openwebui_text_completion(model_id: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-lgos-responses",
        "object": "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
