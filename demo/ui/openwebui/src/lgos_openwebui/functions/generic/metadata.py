"""Build standard Responses metadata from Open WebUI chat state."""

import json
from collections.abc import Collection

from .contracts import (
    CONVERSATION_METADATA_KEY,
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    SETTINGS_METADATA_KEY,
    OpenWebUIMetadata,
)


def _request_metadata(
    metadata: OpenWebUIMetadata,
    *,
    include_runtime_settings: bool = True,
    excluded_runtime_settings: Collection[str] = (),
) -> dict[str, str]:
    request_metadata = (
        _runtime_settings_metadata(metadata, excluded=excluded_runtime_settings)
        if include_runtime_settings
        else {}
    )
    if metadata.chat_id:
        request_metadata[CONVERSATION_METADATA_KEY] = metadata.chat_id
    return request_metadata


def _runtime_settings_metadata(
    metadata: OpenWebUIMetadata, *, excluded: Collection[str] = ()
) -> dict[str, str]:
    if not metadata.chat_variables:
        return {}
    settings = {
        name: value
        for name, value in metadata.chat_variables.items()
        if name not in excluded
    }
    if not settings:
        return {}
    try:
        encoded = json.dumps(
            settings,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        msg = "The selected runtime settings cannot be encoded as JSON."
        raise ValueError(msg) from exc
    if len(encoded) > OPENAI_METADATA_VALUE_MAX_LENGTH:
        msg = "The selected runtime settings exceed the OpenAI metadata value limit."
        raise ValueError(msg)
    return {SETTINGS_METADATA_KEY: encoded}
