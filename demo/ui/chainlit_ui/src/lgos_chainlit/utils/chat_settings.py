"""Translate LGOS model metadata into Chainlit chat settings."""

import json
import logging
from collections.abc import Mapping
from typing import Any, cast

import chainlit as cl
from chainlit.input_widget import InputWidget, Select, Switch, TextInput

from lgos_chainlit.lgos_protocol import (
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    RUNTIME_SETTINGS_METADATA_KEY,
    ModelClientSettings,
    model_client_settings,
)
from lgos_chainlit.utils.clients import discovery_client

logger = logging.getLogger(__name__)
RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY = "lgos_runtime_settings_defaults"


class SettingsTransportError(ValueError):
    """A committed UI setting cannot be represented by the OpenAI request."""


async def configure_chat_settings() -> None:
    """Retrieve the selected model and publish its supported settings."""
    model_id = cast(str, cl.user_session.get("chat_profile"))
    saved = cl.user_session.get("chat_settings")
    candidates = dict(saved) if isinstance(saved, dict) else None
    _store_runtime_settings_defaults(None)
    try:
        model = await discovery_client.models.retrieve(model_id)
    except Exception:
        logger.warning(
            "Runtime settings discovery failed for %s; settings are inactive",
            model_id,
            exc_info=True,
        )
        await cl.ChatSettings([]).refresh()
        return

    client_settings = model_client_settings(model)
    if client_settings is None:
        await cl.ChatSettings([]).send()
        return

    widgets = settings_widgets(client_settings, candidates)
    await cl.ChatSettings(widgets).send()
    _store_runtime_settings_defaults(client_settings.defaults)


def chat_settings_metadata() -> dict[str, str]:
    """Encode the current settings relative to their discovered defaults."""
    defaults = cl.user_session.get(RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY)
    selected = cl.user_session.get("chat_settings")
    return settings_metadata(
        defaults if isinstance(defaults, dict) else None,
        selected if isinstance(selected, dict) else None,
    )


def settings_widgets(
    settings: ModelClientSettings,
    candidates: Mapping[str, Any] | None = None,
) -> list[InputWidget]:
    """Build widgets for direct scalar properties with concrete defaults."""
    properties = settings.json_schema.get("properties")
    if not isinstance(properties, dict):
        return []

    candidates = candidates or {}
    widgets: list[InputWidget] = []
    for name, default in settings.defaults.items():
        widget = _widget_for_property(
            name,
            properties.get(name),
            default,
            candidates,
        )
        if widget is not None:
            widgets.append(widget)
    return widgets


def settings_metadata(
    defaults: Mapping[str, Any] | None,
    values: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Encode changed settings without silently replacing invalid values."""
    if defaults is None or values is None:
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
        raise SettingsTransportError(
            "The selected settings cannot be encoded as JSON."
        ) from exc
    if len(encoded) > OPENAI_METADATA_VALUE_MAX_LENGTH:
        raise SettingsTransportError(
            "The selected settings exceed the OpenAI metadata value limit."
        )

    return {RUNTIME_SETTINGS_METADATA_KEY: encoded}


def _store_runtime_settings_defaults(
    defaults: Mapping[str, object] | None,
) -> None:
    """Store the baseline needed to encode settings on later requests."""
    cl.user_session.set(
        RUNTIME_SETTINGS_DEFAULTS_SESSION_KEY,
        dict(defaults) if defaults is not None else None,
    )


def _widget_for_property(
    name: str,
    schema: Any,
    default: Any,
    candidates: Mapping[str, Any],
) -> InputWidget | None:
    """Build one widget from the small schema subset Chainlit can represent."""
    if not name or not isinstance(schema, dict):
        return None

    label = str(schema.get("title") or name.replace("_", " ").title())
    description = _optional_text(schema.get("description"))
    schema_type = schema.get("type")
    candidate = candidates.get(name)

    if schema_type == "boolean":
        if type(default) is not bool:
            return None
        initial = candidate if type(candidate) is bool else default
        return Switch(
            id=name,
            label=label,
            description=description,
            initial=initial,
        )

    if schema_type != "string" or not isinstance(default, str):
        return None

    if "enum" in schema:
        enum = schema["enum"]
        if (
            not isinstance(enum, list)
            or not enum
            or any(not isinstance(value, str) for value in enum)
            or len(set(enum)) != len(enum)
            or default not in enum
        ):
            return None
        initial = (
            candidate if isinstance(candidate, str) and candidate in enum else default
        )
        return Select(
            id=name,
            label=label,
            description=description,
            values=enum,
            initial_value=initial,
        )

    initial = candidate if isinstance(candidate, str) else default
    return TextInput(
        id=name,
        label=label,
        description=description,
        initial=initial,
    )


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None
