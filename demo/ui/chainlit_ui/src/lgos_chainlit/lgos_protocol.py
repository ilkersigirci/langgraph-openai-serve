"""The narrow LGOS wire contract consumed by the standalone Chainlit client.

This module deliberately duplicates only the public protocol declarations that
the UI needs. It must not import ``langgraph_openai_serve``: the Chainlit image
is an independent OpenAI client, not an LGOS Python application.

Authoritative LGOS sources:

* Model discovery schemas:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
* Graph feature values:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/features.py
* OpenAI metadata limits:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/schemas.py
* Runtime-settings metadata key:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/client_settings.py
* Stream-event opt-in keys:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/utils/events.py
* Client-event schema:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/events.py
* Interrupt tool name:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/utils/interrupts.py
* Interrupt thread metadata key:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/utils.py
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

LGOS_EXTENSION_KEY = "langgraph_openai_serve"
MODEL_EXTENSION_SCHEMA_VERSION = 1
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
STREAM_EVENTS_METADATA_KEY = "langgraph_stream_events"
STREAM_EVENTS_METADATA_VALUE = "v1"
THREAD_METADATA_KEY = "langgraph_thread_id"
INTERRUPT_TOOL_NAME = "langgraph_interrupt"


class GraphFeature(StrEnum):
    """Features advertised for an LGOS model."""

    INTERRUPTS = "interrupts"


class ModelClientSettings(BaseModel):
    """Versioned runtime-settings descriptor advertised for one model."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1] = 1
    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


class LangGraphModelExtension(BaseModel):
    """Versioned LGOS extension returned by detailed model discovery."""

    schema_version: Literal[1] = 1
    features: list[GraphFeature]
    client_settings: ModelClientSettings | None = None


ClientEventType = Literal["status", "progress", "artifact"]


class ClientEventData(BaseModel):
    """A validated public event carried by an LGOS stream extension."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    type: ClientEventType
    namespace: tuple[str, ...] = ()
    data: JsonValue


class ClientEventExtension(BaseModel):
    """The LGOS-specific portion of an OpenAI chat completion chunk."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1]
    event: ClientEventData
