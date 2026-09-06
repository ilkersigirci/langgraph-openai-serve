"""Request models for the supported Chat Completions subset."""

from enum import StrEnum
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
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


class Role(StrEnum):
    """Role options for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallFunction(BaseModel):
    """Model for a tool call function."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Model for a tool call."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


ChatCompletionMessageContent = str | list[ChatCompletionContentPartParam]


class ChatCompletionRequestMessage(BaseModel):
    """Model for a chat completion request message."""

    role: Role
    content: ChatCompletionMessageContent | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    # SDK assistant messages include this as null even for modern tool calls.
    function_call: None = None


class FunctionDefinition(BaseModel):
    """Model for a function definition."""

    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None


class Tool(BaseModel):
    """Model for a tool."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedToolChoiceFunction(BaseModel):
    """Function selected by a named Chat Completions tool choice."""

    name: str


class NamedToolChoice(BaseModel):
    """Named function tool choice accepted by Chat Completions."""

    type: Literal["function"] = "function"
    function: NamedToolChoiceFunction


ChatToolChoice = Literal["none", "auto", "required"] | NamedToolChoice


class ChatCompletionRequest(BaseModel):
    """Model for a chat completion request."""

    model_config = ConfigDict(extra="forbid")

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


class ChatCompletionStreamOptions(BaseModel):
    """Options that affect Chat Completions streaming."""

    include_usage: bool | None = False
