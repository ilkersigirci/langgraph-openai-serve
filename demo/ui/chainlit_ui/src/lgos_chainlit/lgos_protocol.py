"""
The narrow LGOS wire contract consumed by the standalone Chainlit client.

This module deliberately duplicates and decodes only the public protocol pieces
that the UI needs. It must not import ``langgraph_openai_serve``: the Chainlit
image is an independent OpenAI client, not an LGOS Python application.

Authoritative LGOS sources:

* Public protocol names:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/protocol.py
* Model-detail extension schema:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
* Graph feature values:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/features.py
* OpenAI metadata limits:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/metadata.py
* Interrupt tool contract:
  https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/responses/interrupts.py
"""

import logging
from typing import Annotated

from openai.types import Model
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StringConstraints,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

logger = logging.getLogger(__name__)

LGOS_EXTENSION_KEY = "lgos"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
CONVERSATION_METADATA_KEY = "conversation_id"
SETTINGS_METADATA_KEY = "lgos_settings"
INTERRUPT_TOOL_NAME = "lgos_interrupt"
BACKGROUND_FEATURE = "background"
FILE_INPUTS_FEATURE = "file_inputs"
MCP_TOOLS_FEATURE = "mcp_tools"


class ModelClientSettings(BaseModel):
    """Runtime-settings descriptor advertised for one model."""

    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")

    json_schema: dict[str, JsonValue]
    defaults: dict[str, JsonValue]


def _settings_or_none(
    value: object, handler: ValidatorFunctionWrapHandler
) -> ModelClientSettings | None:
    # Malformed settings disable only the settings form, not the model's features.
    try:
        return handler(value)
    except ValidationError:
        logger.warning("Ignoring invalid LGOS runtime settings")
        return None


class LangGraphModelExtension(BaseModel):
    """Forward-compatible LGOS extension returned by model retrieval."""

    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")

    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    features: list[str]
    client_settings: Annotated[
        ModelClientSettings | None, WrapValidator(_settings_or_none)
    ] = None


def model_extension(model: Model) -> LangGraphModelExtension | None:
    """Parse the LGOS extension preserved by the OpenAI SDK."""
    extension = (model.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if extension is None:
        return None
    try:
        return LangGraphModelExtension.model_validate(extension)
    except ValidationError:
        logger.warning("Ignoring invalid LGOS metadata for model %s", model.id)
        return None
