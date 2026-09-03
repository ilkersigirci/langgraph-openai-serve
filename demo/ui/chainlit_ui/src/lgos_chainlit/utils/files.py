"""Upload Chainlit attachments through the OpenAI Files API."""

from pathlib import Path
from typing import cast

from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)

from lgos_chainlit.utils.clients import files_request


async def with_file_parts(
    messages: list[ChatCompletionMessageParam],
    message: object,
) -> list[ChatCompletionMessageParam]:
    """Attach current Chainlit files to the latest OpenAI user message."""
    elements = getattr(message, "elements", None)
    if not isinstance(elements, list) or not elements:
        return messages

    client, provider = files_request()
    parts: list[ChatCompletionContentPartParam] = []
    for element in elements:
        path = getattr(element, "path", None)
        if not isinstance(path, str) or not path:
            continue
        filename = getattr(element, "name", None) or Path(path).name
        content_type = getattr(element, "mime", None) or "application/octet-stream"
        with Path(path).open("rb") as content:
            uploaded = await client.files.create(
                file=(filename, content, content_type),
                purpose="user_data",
                extra_query={"provider": provider} if provider else None,
            )
        parts.append({"type": "file", "file": {"file_id": uploaded.id}})

    if not parts:
        return messages

    if not messages:
        text = getattr(message, "content", None)
        content = (
            [{"type": "text", "text": text}] if isinstance(text, str) and text else []
        )
        return [
            cast(
                "ChatCompletionMessageParam",
                {"role": "user", "content": [*content, *parts]},
            )
        ]

    latest = messages[-1]
    text = latest.get("content")
    content = [{"type": "text", "text": text}] if isinstance(text, str) and text else []
    updated = {**latest, "content": [*content, *parts]}
    return [*messages[:-1], cast("ChatCompletionMessageParam", updated)]
