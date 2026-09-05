"""Build standard Responses metadata from Open WebUI chat state."""

import json
from typing import Any

from .contracts import (
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    RUNTIME_SETTINGS_METADATA_KEY,
    SESSION_ID_METADATA_KEY,
)


def _request_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    request_metadata = _runtime_settings_metadata(metadata)
    chat_id = metadata.get("chat_id")
    if isinstance(chat_id, str) and chat_id:
        request_metadata[SESSION_ID_METADATA_KEY] = chat_id
    return request_metadata


def _runtime_settings_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    values = metadata.get("chat_variables")
    if not isinstance(values, dict) or not values:
        return {}
    try:
        encoded = json.dumps(
            values,
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
    return {RUNTIME_SETTINGS_METADATA_KEY: encoded}
