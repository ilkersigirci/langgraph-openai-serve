"""Resolve OpenAI file IDs into model-ready LangChain content blocks."""

from base64 import b64encode
from collections.abc import Mapping, Sequence
from mimetypes import guess_type
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.content import ContentBlock, create_file_block
from openai import AsyncOpenAI


async def load_file_block(
    client: AsyncOpenAI, file_id: str, *, detail: str = "auto"
) -> ContentBlock | dict[str, Any]:
    """Download one compatible Files API object as model-ready content."""
    metadata = await client.files.retrieve(file_id)
    # The demo Files service returns the text it extracts from a document, and
    # the original upload only when asked for it.
    download = await client.files.content(file_id, extra_query={"raw": "true"})
    content_type = download.response.headers.get(
        "content-type", "application/octet-stream"
    ).partition(";")[0]
    if content_type == "application/octet-stream":
        content_type = guess_type(metadata.filename)[0] or content_type
    encoded = b64encode(await download.aread()).decode("ascii")
    if content_type.startswith("image/"):
        # LangChain sends an image's detail only from this Chat Completions part,
        # not from its standard image block.
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{content_type};base64,{encoded}",
                "detail": detail,
            },
        }
    return create_file_block(
        base64=encoded,
        mime_type=content_type,
        filename=metadata.filename,
    )


def file_id(part: object) -> str | None:
    """Return the Files API ID of an LGOS file or image part, if it has one."""
    if not isinstance(part, Mapping):
        return None
    match part.get("type"):
        case "file":
            file = part.get("file")
            value = file.get("file_id") if isinstance(file, Mapping) else None
        case "image":
            value = part.get("file_id")
        case _:
            value = None
    return value if isinstance(value, str) and value else None


def _detail(part: Mapping[str, Any]) -> str:
    """Return the detail LGOS keeps with an image part, as LangChain does."""
    extras = part.get("extras")
    detail = extras.get("detail") if isinstance(extras, Mapping) else None
    return detail if isinstance(detail, str) else "auto"


async def resolve_file_inputs(
    messages: Sequence[BaseMessage],
    client: AsyncOpenAI,
) -> list[BaseMessage]:
    """Replace file and image ID parts without persisting their bytes in graph state."""
    resolved: list[BaseMessage] = []
    cache: dict[tuple[str, str], ContentBlock | dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, HumanMessage) or not isinstance(
            message.content, list
        ):
            resolved.append(message)
            continue

        content: list[object] = []
        changed = False
        for part in message.content:
            part_file_id = file_id(part)
            if part_file_id is None or not isinstance(part, Mapping):
                content.append(part)
                continue
            key = (part_file_id, _detail(part))
            if key not in cache:
                cache[key] = await load_file_block(client, part_file_id, detail=key[1])
            content.append(cache[key])
            changed = True
        resolved.append(
            message.model_copy(update={"content": content}) if changed else message
        )
    return resolved
