"""Wire-contract values and small models used by the Generic Function."""

from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# These values mirror the public LGOS wire contract. This standalone Open WebUI
# Function must not import the server package:
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/utils/interrupts.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/utils/events.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/chat/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/client_settings.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/features.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/events.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/utils.py
INTERRUPT_TOOL_NAME = "langgraph_interrupt"
ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_CALL_ID_PREFIX = "lgos_ask_"
ASK_USER_MAX_QUESTIONS = 3
ASK_USER_QUESTION_MAX_LENGTH = 500
ASK_USER_REJECTED_OUTPUT = "Error: tool call rejected by user."
INTERRUPT_CANCELLED_MESSAGE = "Interrupt cancelled."
LGOS_EXTENSION_KEY = "langgraph_openai_serve"
CLIENT_EVENTS_FEATURE = "client_events"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
SESSION_ID_METADATA_KEY = "session_id"
RUNTIME_SETTINGS_METADATA_KEY = "langgraph_runtime_settings"
STREAM_EVENTS_METADATA_KEY = "langgraph_stream_events"
STREAM_EVENTS_METADATA_VALUE = "v1"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
NO_CHOICES_MESSAGE = "LangGraph API returned no choices."
CHAT_COMPLETION_REQUEST_FIELDS = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "tools",
    "tool_choice",
)
LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: the configured OpenAI endpoint did not return valid "
    "langgraph_openai_serve model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable. Configure the proxy to pass LGOS /v1 requests "
    "and responses through unchanged."
)
PipeChunk = dict[str, Any]
PipeResponse = AsyncIterator[PipeChunk] | PipeChunk


class InterruptCancelled(Exception):
    """The user cancelled Open WebUI's native interrupt prompt."""


class ChartSeries(BaseModel):
    """One portable series in a chart artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    name: str = Field(min_length=1)
    values: list[float]


class ChartArtifact(BaseModel):
    """The supported LGOS chart artifact payload."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    schema_version: Literal[1]
    id: str = Field(min_length=1)
    kind: Literal["chart"]
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    chart_type: Literal["bar", "line"]
    labels: list[str]
    series: list[ChartSeries]
    x_axis_title: str = Field(min_length=1)
    y_axis_title: str = Field(min_length=1)
    show_legend: bool
