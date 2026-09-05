"""Wire-contract values and small models used by the Generic Function."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# These values mirror the public LGOS wire contract. This standalone Open WebUI
# Function must not import the server package:
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/responses/interrupts.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/metadata.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/client_settings.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/features.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/utils.py
INTERRUPT_TOOL_NAME = "langgraph_interrupt"
DISPLAY_FILE_TOOL_NAME = "display_file"
PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"
ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_CALL_ID_PREFIX = "lgos_ask_"
ASK_USER_MAX_QUESTIONS = 3
ASK_USER_QUESTION_MAX_LENGTH = 500
ASK_USER_REJECTED_OUTPUT = "Error: tool call rejected by user."
INTERRUPT_CANCELLED_MESSAGE = "Interrupt cancelled."
LGOS_EXTENSION_KEY = "langgraph_openai_serve"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
SESSION_ID_METADATA_KEY = "session_id"
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
PipeChunk = str | dict[str, Any]
PipeResponse = AsyncIterator[PipeChunk] | PipeChunk


class InterruptCancelled(Exception):
    """The user cancelled Open WebUI's native interrupt prompt."""


class DisplayFileArguments(BaseModel):
    """Arguments for the client-owned file display function."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    media_type: str = Field(pattern=r"^(?:image/|application/vnd\.plotly\.v1\+json$)")
    title: str = Field(min_length=1)
    alt: str = Field(min_length=1)


class PlotlyFigure(BaseModel):
    """Native figure structure; Plotly.js owns trace and layout semantics."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    data: list[dict[str, JsonValue]]
    layout: dict[str, JsonValue] = Field(default_factory=dict)
    frames: list[dict[str, JsonValue]] = Field(default_factory=list)
