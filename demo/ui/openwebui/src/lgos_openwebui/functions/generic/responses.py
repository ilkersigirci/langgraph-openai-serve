"""Responses API helpers for Open WebUI models."""

from typing import Any

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import TypeAdapter

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


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Open WebUI's text/file transcript into Responses items."""
    items = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system", "developer"}:
            continue
        message_fields = {"role": role}
        if role == "assistant":
            message_fields["phase"] = message.get("phase") or "final_answer"
        if isinstance(content, str):
            items.append({**message_fields, "content": content})
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
        if parts:
            items.append({**message_fields, "content": parts})
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
        if item.type != "message" or item.phase == "commentary":
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


async def _emit_response_sources(response: Response, event_emitter: Any) -> None:
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
