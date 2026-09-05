"""Upload Chainlit attachments through the OpenAI Files API."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from chainlit.config import (
    ChainlitConfigOverrides,
    FeaturesSettings,
    SpontaneousFileUploadFeature,
)
from chainlit.context import context as chainlit_context
from openai.types import Model

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


async def with_response_file_parts(
    input_items: list[dict[str, Any]],
    message: object,
) -> list[dict[str, Any]]:
    """Attach current Chainlit files as native Responses input parts."""
    file_ids = await _upload_file_ids(message)
    if not file_ids:
        return input_items

    user_message_index = next(
        (
            index
            for index in range(len(input_items) - 1, -1, -1)
            if input_items[index].get("role") == "user"
        ),
        None,
    )
    if user_message_index is None:
        return [
            *input_items,
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id} for file_id in file_ids
                ],
            },
        ]

    item = input_items[user_message_index]
    content = item.get("content")
    if isinstance(content, str):
        parts: list[object] = (
            [{"type": "input_text", "text": content}] if content else []
        )
    elif isinstance(content, list):
        parts = list(content)
    else:
        parts = []
    updated = {
        **item,
        "content": [
            *parts,
            *({"type": "input_file", "file_id": file_id} for file_id in file_ids),
        ],
    }
    return [
        *input_items[:user_message_index],
        updated,
        *input_items[user_message_index + 1 :],
    ]


async def _upload_file_ids(message: object) -> list[str]:
    elements = getattr(message, "elements", None)
    if not isinstance(elements, list) or not elements:
        return []
    if not _session_file_upload_enabled():
        msg = "The selected graph does not support file inputs."
        raise ValueError(msg)

    client, provider = files_request()
    file_ids = []
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
                extra_query={"provider": provider},
            )
        file_ids.append(uploaded.id)
    return file_ids


def _session_file_upload_enabled() -> bool:
    """Read the effective chat-profile setting for the current session."""
    session = cast("WebsocketSession", chainlit_context.session)
    upload = session.config.features.spontaneous_file_upload
    return upload is not None and upload.enabled is True
