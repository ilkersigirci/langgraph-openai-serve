"""The narrow LGOS wire contract consumed by the standalone Chainlit client.

This module deliberately duplicates and decodes only the public protocol pieces
that the UI needs. It must not import ``langgraph_openai_serve``: the Chainlit
image is an independent OpenAI client, not an LGOS Python application.

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

import logging
from enum import StrEnum
from typing import Literal

from openai.types import Model
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

logger = logging.getLogger(__name__)

LGOS_EXTENSION_KEY = "langgraph_openai_serve"
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

    schema_version: Literal[1]
    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


class LangGraphModelExtension(BaseModel):
    """Forward-compatible LGOS extension returned by model discovery."""

    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")

    schema_version: Literal[1]
    features: list[str]
    client_settings: JsonValue = None


def model_extension(model: Model) -> LangGraphModelExtension | None:
    """Parse the versioned LGOS extension preserved by the OpenAI SDK."""
    extension = (model.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict):
        return None

    try:
        return LangGraphModelExtension.model_validate(extension)
    except ValidationError:
        logger.warning("Ignoring invalid LGOS metadata for model %s", model.id)
        return None


def model_supports(model: Model, feature: GraphFeature) -> bool:
    """Return whether retrieved model metadata declares an LGOS feature."""
    extension = model_extension(model)
    return extension is not None and feature.value in extension.features


def model_client_settings(model: Model) -> ModelClientSettings | None:
    """Return a supported runtime settings descriptor, when available."""
    extension = model_extension(model)
    if extension is None or extension.client_settings is None:
        return None

    try:
        return ModelClientSettings.model_validate(extension.client_settings)
    except ValidationError:
        logger.warning(
            "Ignoring unsupported LGOS runtime settings for model %s",
            model.id,
        )
        return None


ClientEventType = Literal["status", "progress", "artifact"]


class StatusUpdate(BaseModel):
    """A portable status update rendered by the standalone client."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    description: str = Field(min_length=1)
    done: bool = False
    hidden: bool = False


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
