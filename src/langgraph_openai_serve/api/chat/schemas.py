"""Pydantic models for the OpenAI API.

This module defines Pydantic models that match the OpenAI API request and response formats.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class Role(str, Enum):
    """Role options for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """Model for a function call."""

    name: str
    arguments: str


class ChatMessage(BaseModel):
    """Model for a chat message."""

    role: Role
    content: str | None = None
    name: str | None = None
    function_call: FunctionCall | None = None


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
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] | None = None


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
    messages: list[ChatCompletionRequestMessage]
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
    functions: list[FunctionDefinition] | None = None
    function_call: str | FunctionCall | None = None
    tools: list[Tool] | None = None
    tool_choice: Any | None = None


class ChatCompletionResponseMessage(BaseModel):
    """Model for a chat completion response message."""

    role: Role
    content: str | None = None
    function_call: FunctionCall | None = None
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
class ChatCompletionStreamResponseDelta(BaseModel):
    """Model for a chat completion stream response delta."""

    role: Role | None = None
    content: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] | None = None


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
