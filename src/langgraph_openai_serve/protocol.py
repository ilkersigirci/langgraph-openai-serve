"""Stable names used by the public LGOS protocol extensions."""

from typing import Final

MODEL_EXTENSION_KEY: Final = "lgos"
"""The OpenAI model field that carries a graph's LGOS metadata."""

STATUS_EVENT_TYPE: Final = "lgos.status"
"""The ``type`` of a custom stream event that reports graph progress."""

SETTINGS_METADATA_KEY: Final = "lgos_settings"
"""The request metadata key whose JSON object sets a graph's client settings."""

CONVERSATION_METADATA_KEY: Final = "conversation_id"
"""The request metadata key that groups a client's runs into one conversation."""

INTERRUPT_TOOL_NAME: Final = "lgos_interrupt"
"""The function-call name under which a paused Response exposes each interrupt."""

JSON_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"
"""The ``$schema`` of a graph's published client settings."""

__all__ = [
    "CONVERSATION_METADATA_KEY",
    "INTERRUPT_TOOL_NAME",
    "JSON_SCHEMA_DIALECT",
    "MODEL_EXTENSION_KEY",
    "SETTINGS_METADATA_KEY",
    "STATUS_EVENT_TYPE",
]
