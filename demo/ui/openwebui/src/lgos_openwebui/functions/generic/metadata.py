"""LGOS model extensions and runtime-settings request metadata."""

import json
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from .api import _model_request
from .contracts import (
    LGOS_EXTENSION_KEY,
    LIMITED_FUNCTIONALITY_MESSAGE,
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    RUNTIME_SETTINGS_METADATA_KEY,
    SESSION_ID_METADATA_KEY,
)


def _request_metadata(
    *,
    model: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    request_metadata = _runtime_settings_metadata(model=model, metadata=metadata)
    chat_id = metadata.get("chat_id")
    if isinstance(chat_id, str) and chat_id:
        request_metadata[SESSION_ID_METADATA_KEY] = chat_id
    return request_metadata


def _runtime_settings_metadata(
    *,
    model: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    values = metadata.get("chat_variables")
    if not isinstance(values, dict) or not values:
        return {}

    defaults = _runtime_settings_defaults(model)
    if defaults is None:
        return {}

    changed: dict[str, Any] = {}
    for name, default in defaults.items():
        if name not in values:
            continue
        value = values[name]
        if type(value) is type(default) and value == default:
            continue
        changed[name] = value

    if not changed:
        return {}
    try:
        encoded = json.dumps(
            changed,
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


def _runtime_settings_defaults(model: Any) -> dict[str, Any] | None:
    extension = _model_extension(model)
    if extension is None:
        return None
    settings = extension.get("client_settings")
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        return None
    defaults = settings.get("defaults")
    return defaults if isinstance(defaults, dict) else None


async def _retrieve_model(
    client: AsyncOpenAI,
    model_id: str,
) -> Any:
    """Return model details, or None when retrieval through this endpoint fails."""
    try:
        return await client.models.retrieve(**_model_request(model_id))
    except OpenAIError:
        return None


def _model_extension(model: Any) -> dict[str, Any] | None:
    extension = (getattr(model, "model_extra", None) or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return None
    description = extension.get("description")
    features = extension.get("features")
    if (
        not isinstance(features, list)
        or any(not isinstance(feature, str) for feature in features)
        or not isinstance(description, str)
        or not description.strip()
    ):
        return None
    return extension


def _extension_supports(extension: dict[str, Any] | None, feature: str) -> bool:
    return extension is not None and feature in extension["features"]


async def _emit_limited_functionality_warning(event_emitter: Any) -> None:
    if event_emitter is None:
        return
    await event_emitter(
        {
            "type": "notification",
            "data": {
                "type": "warning",
                "content": LIMITED_FUNCTIONALITY_MESSAGE,
            },
        }
    )
