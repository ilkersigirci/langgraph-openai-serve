"""Validated request models for the supported Responses API subset."""

from typing import Annotated, Literal, TypeAlias

from openai.types.responses import (
    ResponseCustomToolCall as ResponseCustomToolCallInput,
    ResponseCustomToolCallOutput as ResponseCustomToolCallOutputInput,
    ResponseFunctionWebSearch as ResponseWebSearchCallInput,
)
from openai.types.responses.response_output_text import Annotation
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
    """
    A standard OpenAI role message provided as input.

    ``phase`` is accepted for every role and used only for assistant messages.
    See https://developers.openai.com/api/reference/resources/responses.
    """

    role: Literal["user", "assistant", "system", "developer"]
    content: ResponseInputContent
    type: Literal["message"] = "message"
    phase: Literal["commentary", "final_answer"] | None = None


class ResponseOutputTextInput(_ResponsesRequestModel):
    """Plain output text replayed from a previous assistant message."""

    annotations: list[Annotation]
    text: str
    type: Literal["output_text"]
    logprobs: list[JsonValue] | None = None
    # responses.stream().get_final_response() adds this even without a text format.
    parsed: None = None


class ResponseRefusalInput(_ResponsesRequestModel):
    """A model refusal replayed from an assistant message."""

    type: Literal["refusal"]
    refusal: str


class ResponseOutputMessageInput(_ResponsesRequestModel):
    """A terminal assistant output message replayed as input."""

    id: str
    content: Annotated[
        list[ResponseOutputTextInput | ResponseRefusalInput], Field(min_length=1)
    ]
    role: Literal["assistant"]
    status: Literal["completed", "incomplete"]
    type: Literal["message"]
    phase: Literal["commentary", "final_answer"] | None = None


class ResponseFunctionCallInput(_ResponsesRequestModel):
    """A function call replayed from a previous Response."""

    arguments: str
    call_id: str
    name: str
    type: Literal["function_call"] = "function_call"
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
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    caller: None = None
    created_by: str | None = None


ResponseInputItem: TypeAlias = (
    ResponseOutputMessageInput
    | ResponseInputMessage
    | ResponseFunctionCallInput
    | ResponseFunctionCallOutputInput
    | ResponseCustomToolCallInput
    | ResponseCustomToolCallOutputInput
    | ResponseWebSearchCallInput
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
    name: Annotated[str, Field(min_length=1)]
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None


class ResponseCustomTool(_ResponsesRequestModel):
    """Select one registered server tool with the Responses custom-tool shape."""

    type: Literal["custom"]
    name: Annotated[str, Field(min_length=1)]


class ResponseWebSearchTool(_ResponsesRequestModel):
    """Select the graph's OpenAI-compatible web-search capability."""

    type: Literal["web_search"]


class ResponseNamedToolChoice(_ResponsesRequestModel):
    """Require one named function or custom tool."""

    type: Literal["function", "custom"]
    name: str


ResponseToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | ResponseNamedToolChoice
)
ResponseTool: TypeAlias = Annotated[
    ResponseFunctionTool | ResponseCustomTool | ResponseWebSearchTool,
    Field(discriminator="type"),
]


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
    store: bool | None = None
    stream: bool | None = False
    text: ResponseTextConfig | None = None
    tools: list[ResponseTool] | None = None
    tool_choice: ResponseToolChoice | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None

    # Accept OpenAI's no-op values; LGOS has no background or stored-response
    # lifecycle, so the decoder rejects their stateful forms.
    background: bool | None = False
    conversation: JsonValue | None = None
    previous_response_id: str | None = None


__all__ = [
    "ResponseCreateRequest",
    "ResponseCustomTool",
    "ResponseCustomToolCallInput",
    "ResponseCustomToolCallOutputInput",
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
    "ResponseRefusalInput",
    "ResponseTextConfig",
    "ResponseTextFormat",
    "ResponseTool",
    "ResponseToolChoice",
    "ResponseWebSearchCallInput",
    "ResponseWebSearchTool",
]
