"""Stable names used by the public LGOS protocol extensions."""

from typing import Final

MODEL_EXTENSION_KEY: Final = "lgos"
STATUS_EVENT_TYPE: Final = "lgos.status"
SETTINGS_METADATA_KEY: Final = "lgos_settings"
CONVERSATION_METADATA_KEY: Final = "conversation_id"
INTERRUPT_TOOL_NAME: Final = "lgos_interrupt"
JSON_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

__all__ = [
    "CONVERSATION_METADATA_KEY",
    "INTERRUPT_TOOL_NAME",
    "JSON_SCHEMA_DIALECT",
    "MODEL_EXTENSION_KEY",
    "SETTINGS_METADATA_KEY",
    "STATUS_EVENT_TYPE",
]
