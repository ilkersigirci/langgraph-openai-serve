"""Request models for the supported Chat Completions subset."""

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from langgraph_openai_serve.api.metadata import (
    OPENAI_METADATA_MAX_PAIRS,
    MetadataKey,
    MetadataValue,
)


class _ChatRequestModel(BaseModel):
    """Reject fields outside the supported Chat Completions subset."""

    model_config = ConfigDict(extra="forbid")


class Role(StrEnum):
    """Role options for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallFunction(_ChatRequestModel):
    """Model for a tool call function."""

    name: str
    arguments: str


class ToolCall(_ChatRequestModel):
    """Model for a tool call."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatCompletionTextContentPart(_ChatRequestModel):
    """One text part in a Chat Completions message."""

    type: Literal["text"]
    text: str


class ChatCompletionFileReference(_ChatRequestModel):
    """One uploaded file selected by its opaque Files API ID."""

    file_id: str


class ChatCompletionFileContentPart(_ChatRequestModel):
    """One native Chat Completions file-ID content part."""

    type: Literal["file"]
    file: ChatCompletionFileReference


ChatCompletionContentPart: TypeAlias = Annotated[
    ChatCompletionTextContentPart | ChatCompletionFileContentPart,
    Field(discriminator="type"),
]
ChatCompletionMessageContent: TypeAlias = list[ChatCompletionContentPart] | str


class ChatCompletionRequestMessage(_ChatRequestModel):
    """Model for a chat completion request message."""

    role: Role
    content: ChatCompletionMessageContent | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    # SDK assistant messages serialize these nulls even when LGOS does not
    # implement the corresponding output modality or deprecated function call.
    refusal: None = None
    annotations: None = None
    audio: None = None
    function_call: None = None


class FunctionDefinition(_ChatRequestModel):
    """Model for a function definition."""

    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None


class Tool(_ChatRequestModel):
    """Model for a tool."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedToolChoiceFunction(_ChatRequestModel):
    """Function selected by a named Chat Completions tool choice."""

    name: str


class NamedToolChoice(_ChatRequestModel):
    """Named function tool choice accepted by Chat Completions."""

    type: Literal["function"] = "function"
    function: NamedToolChoiceFunction


ChatToolChoice = NamedToolChoice | Literal["none", "auto", "required"]


class ChatCompletionRequest(_ChatRequestModel):
    """Model for a chat completion request."""

    model: str
    messages: list[ChatCompletionRequestMessage] = Field(min_length=1)
    stream: bool | None = False
    stream_options: "ChatCompletionStreamOptions | None" = None
    user: str | None = None
    tools: list[Tool] | None = None
    tool_choice: ChatToolChoice | None = None
    parallel_tool_calls: bool | None = None
    metadata: dict[MetadataKey, MetadataValue] | None = Field(
        default=None,
        max_length=OPENAI_METADATA_MAX_PAIRS,
    )

    @model_validator(mode="after")
    def validate_stream_options(self) -> "ChatCompletionRequest":
        """Allow stream options only for streaming requests."""
        if self.stream_options is not None and not self.stream:
            msg = "stream_options may only be set when stream is true"
            raise ValueError(msg)
        return self


class ChatCompletionStreamOptions(_ChatRequestModel):
    """Options that affect Chat Completions streaming."""

    include_usage: bool | None = False
