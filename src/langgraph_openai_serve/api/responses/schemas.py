"""Validated request models for the supported Responses API subset."""

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from langgraph_openai_serve.api.metadata import (
    OPENAI_METADATA_MAX_PAIRS,
    MetadataKey,
    MetadataValue,
)


class _ResponsesRequestModel(BaseModel):
    """Reject fields outside the supported Responses subset."""

    model_config = ConfigDict(extra="forbid")


class ResponseInputText(_ResponsesRequestModel):
    """One plain-text input content part."""

    type: Literal["input_text"]
    text: str


class ResponseInputFile(_ResponsesRequestModel):
    """One file stored in the configured OpenAI Files service."""

    type: Literal["input_file"]
    file_id: Annotated[str, Field(min_length=1)]


ResponseInputContentPart: TypeAlias = Annotated[
    ResponseInputText | ResponseInputFile,
    Field(discriminator="type"),
]
ResponseInputContent: TypeAlias = (
    str
    | Annotated[
        list[ResponseInputContentPart],
        Field(min_length=1),
    ]
)


class ResponseInputMessage(_ResponsesRequestModel):
    """A user, system, or developer input message."""

    role: Literal["user", "system", "developer"]
    content: ResponseInputContent
    type: Literal["message"] = "message"


class ResponseAssistantInputMessage(_ResponsesRequestModel):
    """A compact assistant message replayed as input."""

    role: Literal["assistant"]
    content: ResponseInputContent
    type: Literal["message"] = "message"
    phase: Literal["commentary", "final_answer"] | None = None


class ResponseOutputTextInput(_ResponsesRequestModel):
    """Plain output text replayed from a previous assistant message."""

    annotations: list[JsonValue]
    text: str
    type: Literal["output_text"]
    logprobs: list[JsonValue] | None = None


class ResponseOutputMessageInput(_ResponsesRequestModel):
    """A completed assistant output message replayed as input."""

    id: str
    content: Annotated[list[ResponseOutputTextInput], Field(min_length=1)]
    role: Literal["assistant"]
    status: Literal["completed"]
    type: Literal["message"]
    phase: Literal["commentary", "final_answer"] | None = None


class ResponseFunctionCallInput(_ResponsesRequestModel):
    """A function call replayed from a previous Response."""

    arguments: str
    call_id: str
    name: str
    type: Literal["function_call"]
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    # Current SDK output models serialize these optional fields as null during
    # full-item replay. Non-null program/namespaced calls are outside this subset.
    caller: None = None
    namespace: None = None


class ResponseFunctionCallOutputInput(_ResponsesRequestModel):
    """Client output for a preceding function call."""

    call_id: str
    output: str
    type: Literal["function_call_output"]
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


ResponseInputItem: TypeAlias = (
    ResponseOutputMessageInput
    | ResponseAssistantInputMessage
    | ResponseInputMessage
    | ResponseFunctionCallInput
    | ResponseFunctionCallOutputInput
)
ResponseInput: TypeAlias = (
    str
    | Annotated[
        list[ResponseInputItem],
        Field(min_length=1),
    ]
)


class ResponseFunctionTool(_ResponsesRequestModel):
    """A client-supplied function available to the graph."""

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None


class ResponseNamedToolChoice(_ResponsesRequestModel):
    """Require one named function tool."""

    type: Literal["function"]
    name: str


ResponseToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | ResponseNamedToolChoice
)


class ResponseTextFormat(_ResponsesRequestModel):
    """The supported plain-text output format."""

    type: Literal["text"]


class ResponseTextConfig(_ResponsesRequestModel):
    """Plain-text response configuration."""

    format: ResponseTextFormat | None = None


class ResponseCreateRequest(_ResponsesRequestModel):
    """The stateless Responses request accepted by LGOS."""

    model: str
    input: ResponseInput
    instructions: str | None = None
    metadata: dict[MetadataKey, MetadataValue] | None = Field(
        default=None,
        max_length=OPENAI_METADATA_MAX_PAIRS,
    )
    store: bool | None = False
    stream: bool | None = False
    text: ResponseTextConfig | None = None
    tools: list[ResponseFunctionTool] | None = None
    tool_choice: ResponseToolChoice | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None

    # Accept OpenAI's no-op values; LGOS has no background or stored-response
    # lifecycle, so the decoder rejects their stateful forms.
    background: bool | None = False
    conversation: JsonValue | None = None
    previous_response_id: str | None = None


__all__ = [
    "ResponseAssistantInputMessage",
    "ResponseCreateRequest",
    "ResponseFunctionCallInput",
    "ResponseFunctionCallOutputInput",
    "ResponseFunctionTool",
    "ResponseInputFile",
    "ResponseInputItem",
    "ResponseInputMessage",
    "ResponseInputText",
    "ResponseNamedToolChoice",
    "ResponseOutputMessageInput",
    "ResponseOutputTextInput",
    "ResponseTextConfig",
    "ResponseTextFormat",
    "ResponseToolChoice",
]
