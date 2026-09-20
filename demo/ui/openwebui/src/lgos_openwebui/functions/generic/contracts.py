"""Wire-contract values and small models used by the Generic Function."""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Literal, Self

from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
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
# https://github.com/ilkersigirci/langgraph-openai-serve/blob/main/src/langgraph_openai_serve/graph/utils.py
INTERRUPT_TOOL_NAME = "lgos_interrupt"
DISPLAY_FILE_TOOL_NAME = "display_file"
PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"
ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_CALL_ID_PREFIX = "lgos_ask_"
ASK_USER_MAX_QUESTIONS = 3
ASK_USER_QUESTION_MAX_LENGTH = 500
ASK_USER_REJECTED_OUTPUT = "Error: tool call rejected by user."
INTERRUPT_CANCELLED_MESSAGE = "Interrupt cancelled."
LGOS_EXTENSION_KEY = "lgos"
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
CONVERSATION_METADATA_KEY = "conversation_id"
SETTINGS_METADATA_KEY = "lgos_settings"
RUN_METADATA_KEY = "lgos_run_id"
LGOS_MODEL_OWNER = "langgraph-openai-serve"
SERVER_TOOL_MODEL_NAME = "server-tool"
ADVANCED_GRAPH_MODEL_NAME = "advanced-graph"
PERSISTENT_PLOT_MODEL_NAME = "persistent-plot-agent"
PACKAGE_VERSION_TOOL_NAME = "lgos_package_version"
WEB_SEARCH_TOOL_NAME = "web_search"
BACKGROUND_SETTING_NAME = "lgos_background"
PipeChunk = str | dict[str, Any]
PipeResponse = AsyncIterator[PipeChunk] | PipeChunk
OpenWebUIEventEmitter = Callable[[dict[str, Any]], Awaitable[object]]


class OpenWebUIHostModel(BaseModel):
    """Validated projection of an additive Open WebUI-owned object."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class OpenWebUIStoredFileMetadata(OpenWebUIHostModel):
    content_type: str | None = None


class OpenWebUIStoredFile(OpenWebUIHostModel):
    path: str | None = None
    filename: str | None = None
    meta: OpenWebUIStoredFileMetadata | None = None


class OpenWebUIFile(OpenWebUIHostModel):
    id: str | None = None
    type: str | None = None
    name: str | None = None
    content_type: str | None = None
    file: OpenWebUIStoredFile | None = None


class OpenWebUIMessageFunction(OpenWebUIHostModel):
    name: str | None = None
    arguments: str | None = None


class OpenWebUIMessageToolCall(OpenWebUIHostModel):
    id: str | None = None
    function: OpenWebUIMessageFunction | None = None


class OpenWebUIMessage(OpenWebUIHostModel):
    role: str | None = None
    content: JsonValue = None
    phase: str | None = None
    tool_calls: list[OpenWebUIMessageToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class OpenWebUIBody(OpenWebUIHostModel):
    model: str = Field(min_length=1)
    messages: list[OpenWebUIMessage]
    stream: bool = False

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        _, separator, model_id = value.partition(".")
        if not separator or not model_id:
            msg = "Open WebUI did not provide a valid model ID."
            raise ValueError(msg)
        return value

    @property
    def model_id(self) -> str:
        return self.model.partition(".")[2]


class OpenWebUIUserMessage(OpenWebUIHostModel):
    files: list[OpenWebUIFile] = Field(default_factory=list)


class OpenWebUIChatVariableField(OpenWebUIHostModel):
    key: str | None = None


class OpenWebUIMetadata(OpenWebUIHostModel):
    chat_id: str | None = None
    chat_variables: dict[str, JsonValue] = Field(default_factory=dict)
    model_chat_variables: list[OpenWebUIChatVariableField] = Field(
        default_factory=list,
        validation_alias=AliasPath(
            "model", "info", "meta", "chat_variables_schema", "fields"
        ),
    )
    user_message: OpenWebUIUserMessage | None = None

    def supports_chat_variable(self, key: str) -> bool:
        return any(field.key == key for field in self.model_chat_variables)


class OpenWebUIUser(OpenWebUIHostModel):
    id: str | None = None


class OpenWebUIToolSpec(OpenWebUIHostModel):
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool = False


class OpenWebUIMCPTool(OpenWebUIHostModel):
    type: Literal["mcp"]
    spec: OpenWebUIToolSpec


class OpenWebUIInvocation(OpenWebUIHostModel):
    """The complete validated subset of one Open WebUI Pipe invocation."""

    body: OpenWebUIBody
    metadata: OpenWebUIMetadata
    user: OpenWebUIUser
    files: list[OpenWebUIFile]
    mcp_tools: dict[str, OpenWebUIMCPTool]

    @classmethod
    def from_host(
        cls,
        *,
        body: object,
        metadata: object,
        user: object,
        files: object,
        tools: object,
    ) -> Self:
        if tools is None:
            raw_tools: Mapping[object, object] = {}
        elif isinstance(tools, Mapping):
            raw_tools = tools
        else:
            msg = "Open WebUI provided invalid Function arguments."
            raise ValueError(msg)

        # Open WebUI may inject unrelated built-in and client tools. Only MCP
        # declarations participate in this Function's Responses request.
        mcp_tools = {
            name: value
            for name, value in raw_tools.items()
            if isinstance(name, str)
            and isinstance(value, Mapping)
            and value.get("type") == "mcp"
        }
        try:
            return cls.model_validate(
                {
                    "body": body,
                    "metadata": metadata if metadata is not None else {},
                    "user": user if user is not None else {},
                    "files": files if files is not None else [],
                    "mcp_tools": mcp_tools,
                }
            )
        except ValidationError as exc:
            msg = "Open WebUI provided invalid Function arguments."
            raise ValueError(msg) from exc


def is_server_tool_model(model_id: str) -> bool:
    """Return whether a model is the fixed server-tool showcase."""
    return model_id.rsplit("/", 1)[-1] == SERVER_TOOL_MODEL_NAME


def supports_web_search(model_id: str) -> bool:
    """Return whether the demo UI may request LGOS server-side web search."""
    return model_id.rsplit("/", 1)[-1] in {
        ADVANCED_GRAPH_MODEL_NAME,
        SERVER_TOOL_MODEL_NAME,
    }


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
