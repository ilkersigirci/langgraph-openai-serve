"""Resolve OpenAI file IDs into model-ready LangChain content blocks."""

from base64 import b64encode
from collections.abc import Mapping, Sequence
from mimetypes import guess_type

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.content import (
    ContentBlock,
    create_file_block,
    create_image_block,
)
from openai import AsyncOpenAI


async def load_file_block(client: AsyncOpenAI, file_id: str) -> ContentBlock:
    """Download one compatible Files API object as a LangChain content block."""
    metadata = await client.files.retrieve(file_id)
    download = await client.files.content(file_id)
    content_type = download.response.headers.get(
        "content-type", "application/octet-stream"
    ).partition(";")[0]
    if content_type == "application/octet-stream":
        content_type = guess_type(metadata.filename)[0] or content_type
    encoded = b64encode(await download.aread()).decode("ascii")
    if content_type.startswith("image/"):
        return create_image_block(base64=encoded, mime_type=content_type)
    return create_file_block(
        base64=encoded,
        mime_type=content_type,
        filename=metadata.filename,
    )


async def resolve_file_inputs(
    messages: Sequence[BaseMessage],
    client: AsyncOpenAI,
) -> list[BaseMessage]:
    """Replace normalized file-ID parts without persisting file bytes in graph state."""
    resolved: list[BaseMessage] = []
    cache: dict[str, ContentBlock] = {}
    for message in messages:
        if not isinstance(message, HumanMessage) or not isinstance(
            message.content, list
        ):
            resolved.append(message)
            continue

        content: list[object] = []
        changed = False
        for part in message.content:
            file = part.get("file") if isinstance(part, Mapping) else None
            file_id = file.get("file_id") if isinstance(file, Mapping) else None
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "file"
                or not isinstance(file_id, str)
            ):
                content.append(part)
                continue
            if file_id not in cache:
                cache[file_id] = await load_file_block(client, file_id)
            content.append(cache[file_id])
            changed = True
        resolved.append(
            message.model_copy(update={"content": content}) if changed else message
        )
    return resolved


__all__ = ["load_file_block", "resolve_file_inputs"]
