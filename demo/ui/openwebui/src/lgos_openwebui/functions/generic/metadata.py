"""Build standard Responses metadata from Open WebUI chat state."""

import json
from typing import Any

from .contracts import (
    CONVERSATION_METADATA_KEY,
    CURRENT_TIME_TOOL_NAME,
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    SETTINGS_METADATA_KEY,
    WEB_SEARCH_TOOL_NAME,
)


def _request_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    request_metadata = _runtime_settings_metadata(metadata)
    chat_id = metadata.get("chat_id")
    if isinstance(chat_id, str) and chat_id:
        request_metadata[CONVERSATION_METADATA_KEY] = chat_id
    return request_metadata


def _runtime_settings_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    values = metadata.get("chat_variables")
    if not isinstance(values, dict):
        return {}
    # Tool controls share Open WebUI's chat variables with graph settings.
    settings = {
        key: value
        for key, value in values.items()
        if key not in (CURRENT_TIME_TOOL_NAME, WEB_SEARCH_TOOL_NAME)
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
