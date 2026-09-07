"""Stable names used by the public LGOS protocol extensions."""

from typing import Final

MODEL_EXTENSION_KEY: Final = "lgos"
MODEL_EXTENSION_SCHEMA_VERSION: Final = 1
CLIENT_SETTINGS_SCHEMA_VERSION: Final = 1
CLIENT_EVENT_TYPE: Final = "lgos.client_event"
CLIENT_EVENT_SCHEMA_VERSION: Final = 1
SETTINGS_METADATA_KEY: Final = "lgos_settings"
CONVERSATION_METADATA_KEY: Final = "conversation_id"
RUN_METADATA_KEY: Final = "lgos_run_id"
INTERRUPT_TOOL_NAME: Final = "lgos_interrupt"
JSON_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

__all__ = [
    "CLIENT_EVENT_SCHEMA_VERSION",
    "CLIENT_EVENT_TYPE",
    "CLIENT_SETTINGS_SCHEMA_VERSION",
    "CONVERSATION_METADATA_KEY",
    "INTERRUPT_TOOL_NAME",
    "JSON_SCHEMA_DIALECT",
    "MODEL_EXTENSION_KEY",
    "MODEL_EXTENSION_SCHEMA_VERSION",
    "RUN_METADATA_KEY",
    "SETTINGS_METADATA_KEY",
]
