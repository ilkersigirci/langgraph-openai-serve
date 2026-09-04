"""Upload Chainlit attachments through the OpenAI Files API."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from chainlit.config import (
    ChainlitConfigOverrides,
    FeaturesSettings,
    SpontaneousFileUploadFeature,
)
from chainlit.context import context as chainlit_context
from openai.types import Model
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)

from lgos_chainlit.lgos_protocol import GraphFeature, model_supports
from lgos_chainlit.utils.clients import files_request

if TYPE_CHECKING:
    from chainlit.session import WebsocketSession


def file_upload_overrides(model: Model | None) -> ChainlitConfigOverrides:
    """Enable Chainlit's attachment control only for file-aware graphs."""
    return ChainlitConfigOverrides(
        features=FeaturesSettings(
            spontaneous_file_upload=SpontaneousFileUploadFeature(
                enabled=(
                    model is not None
                    and model_supports(model, GraphFeature.FILE_INPUTS)
                )
            )
        )
    )


async def with_file_parts(
    messages: list[ChatCompletionMessageParam],
    message: object,
) -> list[ChatCompletionMessageParam]:
    """Attach current Chainlit files to the latest OpenAI user message."""
    elements = getattr(message, "elements", None)
    if not isinstance(elements, list) or not elements:
        return messages
    if not _session_file_upload_enabled():
        msg = "The selected graph does not support file inputs."
        raise ValueError(msg)

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


def _session_file_upload_enabled() -> bool:
    """Read the effective chat-profile setting for the current session."""
    session = cast("WebsocketSession", chainlit_context.session)
    upload = session.config.features.spontaneous_file_upload
    return upload is not None and upload.enabled is True
