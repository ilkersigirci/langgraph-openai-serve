"""Translate LGOS model metadata into a small set of Chainlit settings."""

import json
import logging
from collections.abc import Mapping
from typing import Any, cast

from chainlit.input_widget import InputWidget, Select, Switch, TextInput
from openai.types import Model
from pydantic import ValidationError

from lgos_chainlit.lgos_protocol import (
    LGOS_EXTENSION_KEY,
    MODEL_EXTENSION_SCHEMA_VERSION,
    OPENAI_METADATA_VALUE_MAX_LENGTH,
    RUNTIME_SETTINGS_METADATA_KEY,
    GraphFeature,
    LangGraphModelExtension,
    ModelClientSettings,
)

logger = logging.getLogger(__name__)
COMMON_SCHEMA_KEYS = {"default", "description", "title", "type"}
STRING_SCHEMA_KEYS = COMMON_SCHEMA_KEYS | {"enum", "maxLength", "minLength"}


class SettingsTransportError(ValueError):
    """A committed UI setting cannot be represented by the OpenAI request."""


def model_extension(model: Model) -> LangGraphModelExtension | None:
    """Parse the versioned LGOS extension preserved by the OpenAI SDK."""
    extension = (model.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if (
        not isinstance(extension, dict)
        or extension.get("schema_version") != MODEL_EXTENSION_SCHEMA_VERSION
    ):
        return None

    raw_features = extension.get("features")
    if not isinstance(raw_features, list):
        logger.warning("Ignoring invalid LGOS metadata for model %s", model.id)
        return None
    features: list[GraphFeature] = []
    for value in raw_features:
        try:
            features.append(GraphFeature(value))
        except (TypeError, ValueError):
            logger.debug("Ignoring unknown LGOS feature %r", value)

    client_settings: ModelClientSettings | None = None
    raw_client_settings = extension.get("client_settings")
    if raw_client_settings is not None:
        try:
            client_settings = ModelClientSettings.model_validate(raw_client_settings)
        except ValidationError:
            logger.warning(
                "Ignoring unsupported LGOS runtime settings for model %s",
                model.id,
            )

    try:
        return LangGraphModelExtension(
            features=features,
            client_settings=client_settings,
        )
    except ValidationError:
        logger.warning("Ignoring unsupported LGOS metadata for model %s", model.id)
        return None


def model_supports(model: Model, feature: GraphFeature) -> bool:
    """Return whether retrieved model metadata declares an LGOS feature."""
    extension = model_extension(model)
    return extension is not None and feature in extension.features


def model_client_settings(model: Model) -> ModelClientSettings | None:
    """Return a supported runtime settings descriptor, when available."""
    extension = model_extension(model)
    return extension.client_settings if extension is not None else None


def restore_setting_values(
    settings: ModelClientSettings,
    candidates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize persisted scalar values for restoring Chainlit widgets."""
    candidates = candidates or {}
    values: dict[str, Any] = {}
    for name in settings.defaults:
        schema = property_schema(settings, name)
        if schema is None:
            continue
        candidate = candidates.get(name)
        values[name] = (
            candidate
            if name in candidates and _valid_scalar(candidate, schema)
            else settings.defaults[name]
        )
    return values


def settings_widgets(
    settings: ModelClientSettings,
    candidates: Mapping[str, Any] | None = None,
) -> tuple[list[InputWidget], dict[str, Any]]:
    """Build widgets for direct scalar properties with concrete defaults."""
    values = restore_setting_values(settings, candidates)
    widgets = [
        _widget_for(name, schema, values[name])
        for name in settings.defaults
        if (schema := property_schema(settings, name)) is not None
    ]
    return widgets, values


def settings_metadata(
    settings: ModelClientSettings | None,
    values: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Encode changed settings without silently replacing invalid values."""
    if settings is None or values is None:
        return {}

    changed: dict[str, Any] = {}
    for name, default in settings.defaults.items():
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


def property_schema(
    settings: ModelClientSettings,
    name: str,
) -> dict[str, Any] | None:
    """Return a directly declared scalar property the demo can render exactly."""
    properties = settings.json_schema.get("properties")
    if not isinstance(properties, dict):
        return None
    schema = properties.get(name)
    if not isinstance(schema, dict) or any(
        keyword in schema for keyword in ("$ref", "allOf", "anyOf", "oneOf")
    ):
        return None
    if name not in settings.defaults:
        return None
    return schema if _supported_schema(schema, settings.defaults[name]) else None


def _supported_schema(schema: Mapping[str, Any], default: Any) -> bool:
    schema_type = schema.get("type")
    if schema_type == "boolean":
        return set(schema) <= COMMON_SCHEMA_KEYS and _valid_scalar(default, schema)

    if schema_type == "string":
        if not set(schema) <= STRING_SCHEMA_KEYS:
            return False
        enum = schema.get("enum")
        if enum is not None and (
            not isinstance(enum, list)
            or not enum
            or any(not isinstance(value, str) for value in enum)
            or any(not _valid_scalar(value, schema) for value in enum)
        ):
            return False
        return _valid_scalar(default, schema)

    return False


def _valid_scalar(value: Any, schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "string":
        if not isinstance(value, str):
            return False
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            return False
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        return not (
            (isinstance(minimum, int) and len(value) < minimum)
            or (isinstance(maximum, int) and len(value) > maximum)
        )
    return False


def _widget_for(
    name: str,
    schema: Mapping[str, Any],
    initial: Any,
) -> InputWidget:
    label = str(schema.get("title") or name.replace("_", " ").title())
    description = _optional_text(schema.get("description"))
    schema_type = schema["type"]
    enum = schema.get("enum")
    if schema_type == "boolean":
        return Switch(
            id=name,
            label=label,
            description=description,
            initial=cast(bool, initial),
        )
    if schema_type == "string" and isinstance(enum, list):
        return Select(
            id=name,
            label=label,
            description=description,
            values=cast(list[str], enum),
            initial_value=cast(str, initial),
        )
    return TextInput(
        id=name,
        label=label,
        description=description,
        initial=cast(str, initial),
    )


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None
