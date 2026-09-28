"""Wire-contract values and small models used by the Generic Function."""

import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol

from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
)

# These values mirror the public LGOS wire contract. This standalone Open WebUI
# Function must not import the server package:
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/protocol.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/models/schemas.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/responses/interrupts.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/api/metadata.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/client_settings.py
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/features.py
INTERRUPT_TOOL_NAME = "lgos_interrupt"
DISPLAY_FILE_TOOL_NAME = "display_file"
PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"
ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_CALL_ID_PREFIX = "lgos_ask_"
# Open WebUI's ask_user limits. It truncates longer text instead of rejecting it.
ASK_USER_MAX_QUESTIONS = 3
ASK_USER_QUESTION_ID_MAX_LENGTH = 64
ASK_USER_QUESTION_MAX_LENGTH = 500
ASK_USER_LABEL_MAX_LENGTH = 80
ASK_USER_REJECTED_OUTPUT = "Error: tool call rejected by user."
INTERRUPT_CANCELLED_MESSAGE = "Interrupt cancelled."
LGOS_EXTENSION_KEY = "lgos"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
CONVERSATION_METADATA_KEY = "conversation_id"
SETTINGS_METADATA_KEY = "lgos_settings"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
PERSISTENT_PLOT_MODEL_NAME = "persistent-plot-agent"
PACKAGE_VERSION_TOOL_NAME = "lgos_package_version"
WEB_SEARCH_TOOL_NAME = "web_search"
BACKGROUND_SETTING_NAME = "lgos_background"
# Chat Variables the Pipe consumes itself; they never become graph settings.
PIPE_SETTING_NAMES = frozenset(
    {BACKGROUND_SETTING_NAME, PACKAGE_VERSION_TOOL_NAME, WEB_SEARCH_TOOL_NAME}
)
INTEGER_TEXT = re.compile(r"-?\d+")
# Open WebUI prepends the rendered declarations to the first system message.
# Form values are single-line, so these newline-anchored lines bound them.
CHAT_VARIABLES_DECLARATION_START = "<lgos-chat-variables>\n"
CHAT_VARIABLES_DECLARATION_END = "\n</lgos-chat-variables>"
PipeChunk = str | dict[str, Any]
PipeResponse = AsyncIterator[PipeChunk] | PipeChunk
OpenWebUIEventEmitter = Callable[[dict[str, Any]], Awaitable[object]]


class OpenWebUIRequest(Protocol):
    """The parts of Open WebUI's Starlette request this Function uses."""

    @property
    def app(self) -> Callable[..., Awaitable[None]]: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


class OpenWebUIFile(BaseModel):
    id: str = ""
    type: str | None = None
    name: str = ""


class OpenWebUIMessageFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class OpenWebUIMessageToolCall(BaseModel):
    id: str | None = None
    function: OpenWebUIMessageFunction | None = None


class OpenWebUIMessage(BaseModel):
    role: str | None = None
    content: JsonValue = None
    phase: str | None = None
    tool_calls: list[OpenWebUIMessageToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class OpenWebUIBody(BaseModel):
    model: str
    messages: list[OpenWebUIMessage]

    @field_validator("messages")
    @classmethod
    def remove_chat_variable_declarations(
        cls, messages: list[OpenWebUIMessage]
    ) -> list[OpenWebUIMessage]:
        """Keep UI settings out of the graph prompt; they travel as metadata."""
        if not messages or messages[0].role != "system":
            return messages
        system = messages[0]
        if not isinstance(system.content, str):
            return messages
        content = system.content
        # Each native tool-loop continuation prepends another copy.
        while content.startswith(CHAT_VARIABLES_DECLARATION_START):
            end = content.find(CHAT_VARIABLES_DECLARATION_END)
            if end == -1:
                break
            end += len(CHAT_VARIABLES_DECLARATION_END)
            content = content[end:].removeprefix("\n")
        if content == system.content:
            return messages
        return [system.model_copy(update={"content": content}), *messages[1:]]

    @property
    def model_id(self) -> str:
        """Return the graph model from Open WebUI's ``<pipe>.<model>`` ID."""
        return self.model.partition(".")[2]


class OpenWebUIUserMessage(BaseModel):
    files: list[OpenWebUIFile] = Field(default_factory=list)


class OpenWebUIChatVariableField(BaseModel):
    key: str | None = None
    type: str | None = None


class OpenWebUIMetadata(BaseModel):
    chat_id: str | None = None
    # Validated before chat_variables, which keep only these declared fields.
    model_chat_variables: list[OpenWebUIChatVariableField] = Field(
        default_factory=list,
        validation_alias=AliasPath(
            "model", "info", "meta", "chat_variables_schema", "fields"
        ),
    )
    chat_variables: dict[str, JsonValue] = Field(default_factory=dict)
    # Complete runtime settings added by a settings Filter such as UserValves.
    lgos_settings: dict[str, JsonValue] | None = None
    user_message: OpenWebUIUserMessage | None = None

    @field_validator("chat_variables")
    @classmethod
    def declared_chat_variables(
        cls, values: dict[str, JsonValue], info: ValidationInfo
    ) -> dict[str, JsonValue]:
        """Keep the selected model's variables and restore their declared types."""
        field_types = {
            field.key: field.type for field in info.data.get("model_chat_variables", [])
        }
        typed: dict[str, JsonValue] = {}
        for key, value in values.items():
            # A chat keeps the variables of every model it has used. Open WebUI
            # treats empty values as unset, so LGOS applies its defaults.
            if key not in field_types or value is None or value == "":
                continue
            field_type = field_types[key]
            if field_type == "checkbox":
                # The form checks the box only for these two values.
                typed[key] = value is True or value == "true"
            elif (
                field_type == "number"
                and isinstance(value, str)
                and INTEGER_TEXT.fullmatch(value)
            ):
                typed[key] = int(value)
            else:
                typed[key] = value
        return typed


class OpenWebUIToolSpec(BaseModel):
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool = False


def supports_display_file(model_id: str) -> bool:
    """Return whether the selected demo graph publishes displayable files."""
    return model_id.rsplit("/", 1)[-1] == PERSISTENT_PLOT_MODEL_NAME


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
