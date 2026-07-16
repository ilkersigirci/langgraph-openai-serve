"""Pydantic models for the OpenAI API.

This module defines Pydantic models that match the OpenAI API request and response formats.
"""

from enum import Enum
from typing import Any, Literal

from openai.types.chat.chat_completion_message import Annotation
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)


def _reject_legacy_fields(
    data: Any,
    replacements: dict[str, str],
    *,
    title: str,
) -> Any:
    if not isinstance(data, dict):
        return data

    for field, replacement in replacements.items():
        if field in data:
            error = ValueError(
                f"'{field}' is not supported; use '{replacement}' instead."
            )
            raise ValidationError.from_exception_data(
                title,
                [
                    {
                        "type": "value_error",
                        "loc": (field,),
                        "input": data[field],
                        "ctx": {"error": error},
                    }
                ],
            )
    return data


class Role(str, Enum):
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


class ChatCompletionRequestMessage(BaseModel):
    """Model for a chat completion request message."""

    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, data: Any) -> Any:
        """Reject the deprecated singular Chat Completions function call."""
        return _reject_legacy_fields(
            data,
            {"function_call": "tool_calls"},
            title=cls.__name__,
        )


class FunctionDefinition(BaseModel):
    """Model for a function definition."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolFunction(BaseModel):
    """Model for a tool function."""

    function: FunctionDefinition


class Tool(BaseModel):
    """Model for a tool."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """Model for a chat completion request."""

    model: str
    messages: list[ChatCompletionRequestMessage] = Field(min_length=1)
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[Tool] | None = None
    tool_choice: Any | None = None
    metadata: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, data: Any) -> Any:
        """Reject deprecated Chat Completions function parameters."""
        return _reject_legacy_fields(
            data,
            {
                "function_call": "tool_choice",
                "functions": "tools",
            },
            title=cls.__name__,
        )


class ChatCompletionResponseMessage(BaseModel):
    """Model for a chat completion response message."""

    role: Role
    content: str | None = None
    annotations: list[Annotation] | None = None
    tool_calls: list[ToolCall] | None = None


class UsageInfo(BaseModel):
    """Model for usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseChoice(BaseModel):
    """Model for a chat completion response choice."""

    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Model for a chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo | None = None


# Stream models
class ChatCompletionStreamToolCallFunction(BaseModel):
    """Model for a streaming tool call function delta."""

    name: str | None = None
    arguments: str | None = None


class ChatCompletionStreamToolCall(BaseModel):
    """Model for a streaming tool call delta."""

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: ChatCompletionStreamToolCallFunction | None = None


class ChatCompletionStreamResponseDelta(BaseModel):
    """Model for a chat completion stream response delta."""

    role: Role | None = None
    content: str | None = None
    tool_calls: list[ChatCompletionStreamToolCall] | None = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """Model for a chat completion stream response choice."""

    index: int
    delta: ChatCompletionStreamResponseDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """Model for a chat completion stream response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
