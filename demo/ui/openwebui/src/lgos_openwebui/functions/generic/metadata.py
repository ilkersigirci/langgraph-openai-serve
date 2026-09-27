"""Build standard Responses metadata from Open WebUI chat state."""

import json

from .contracts import (
    CONVERSATION_METADATA_KEY,
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    PIPE_SETTING_NAMES,
    SETTINGS_METADATA_KEY,
    OpenWebUIMetadata,
)


def _request_metadata(metadata: OpenWebUIMetadata) -> dict[str, str]:
    request_metadata = {}
    settings = (
        metadata.lgos_settings
        if metadata.lgos_settings is not None
        else {
            name: value
            for name, value in metadata.chat_variables.items()
            if name not in PIPE_SETTING_NAMES
        }
    )
    if settings:
        encoded = json.dumps(
            settings,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(encoded) > OPENAI_METADATA_VALUE_MAX_LENGTH:
            msg = (
                "The selected runtime settings exceed the OpenAI metadata value limit."
            )
            raise ValueError(msg)
        request_metadata[SETTINGS_METADATA_KEY] = encoded
    if metadata.chat_id:
        request_metadata[CONVERSATION_METADATA_KEY] = metadata.chat_id
    return request_metadata
