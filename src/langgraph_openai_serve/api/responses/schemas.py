"""Validated request models for the supported Responses API subset."""

from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, Field, JsonValue, Tag

from langgraph_openai_serve.api.metadata import (
    OPENAI_METADATA_MAX_PAIRS,
    MetadataKey,
    MetadataValue,
)


class _ResponsesRequestModel(BaseModel):
    """Reject fields outside the supported Responses subset."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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
    Annotated[
        list[ResponseInputContentPart],
        Field(min_length=1),
    ]
    | str
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


class ResponseURLCitationInput(_ResponsesRequestModel):
    """A URL citation replayed with assistant output text."""

    end_index: int
    start_index: int
    title: str
    type: Literal["url_citation"]
    url: str


class ResponseOutputTextInput(_ResponsesRequestModel):
    """Plain output text replayed from a previous assistant message."""

    annotations: list[ResponseURLCitationInput]
    text: str
    type: Literal["output_text"]
    logprobs: Annotated[list[JsonValue], Field(max_length=0)] | None = None
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
        list[
            Annotated[
                ResponseOutputTextInput | ResponseRefusalInput,
                Field(discriminator="type"),
            ]
        ],
        Field(min_length=1),
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
    # OpenAI v3 adds this field when SDK response objects are replayed as input.
    async_: bool | None = Field(default=None, alias="async")
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


class ResponseCustomToolCallInput(_ResponsesRequestModel):
    """A custom-tool call replayed from a previous Response."""

    call_id: str
    input: str
    name: str
    type: Literal["custom_tool_call"] = "custom_tool_call"
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    async_: bool | None = Field(default=None, alias="async")
    caller: None = None
    namespace: None = None


class ResponseCustomToolCallOutputInput(_ResponsesRequestModel):
    """A string result replayed for a preceding custom-tool call."""

    call_id: str
    output: str
    type: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    caller: None = None
    created_by: str | None = None


class ResponseWebSearchActionInput(_ResponsesRequestModel):
    """The query action produced by LGOS's supported web-search tool."""

    type: Literal["search"]
    query: str
    # The locked SDK includes these nullable fields on serialized output items.
    queries: None = None
    sources: None = None


class ResponseWebSearchCallInput(_ResponsesRequestModel):
    """A web-search call replayed from a previous Response."""

    id: str
    action: ResponseWebSearchActionInput
    status: Literal["in_progress", "searching", "completed", "failed"]
    type: Literal["web_search_call"]


def _response_input_item_type(value: object) -> str | None:
    """Discriminate input and output messages that share ``type='message'``."""
    if isinstance(value, ResponseOutputMessageInput):
        item_type = "output_message"
    elif isinstance(value, ResponseInputMessage):
        item_type = "input_message"
    elif isinstance(
        value,
        (
            ResponseFunctionCallInput,
            ResponseFunctionCallOutputInput,
            ResponseCustomToolCallInput,
            ResponseCustomToolCallOutputInput,
            ResponseWebSearchCallInput,
        ),
    ):
        item_type = value.type
    elif isinstance(value, Mapping):
        discriminator = value.get("type")
        if discriminator == "message":
            item_type = (
                "output_message"
                if "id" in value or "status" in value
                else "input_message"
            )
        elif discriminator is None and "role" in value:
            item_type = "input_message"
        else:
            item_type = discriminator if isinstance(discriminator, str) else None
    else:
        item_type = None
    return item_type


ResponseInputItem: TypeAlias = Annotated[
    Annotated[ResponseOutputMessageInput, Tag("output_message")]
    | Annotated[ResponseInputMessage, Tag("input_message")]
    | Annotated[ResponseFunctionCallInput, Tag("function_call")]
    | Annotated[ResponseFunctionCallOutputInput, Tag("function_call_output")]
    | Annotated[ResponseCustomToolCallInput, Tag("custom_tool_call")]
    | Annotated[
        ResponseCustomToolCallOutputInput,
        Tag("custom_tool_call_output"),
    ]
    | Annotated[ResponseWebSearchCallInput, Tag("web_search_call")],
    Discriminator(_response_input_item_type),
]
ResponseInput: TypeAlias = (
    Annotated[
        list[ResponseInputItem],
        Field(min_length=1),
    ]
    | str
)


class ResponseFunctionTool(_ResponsesRequestModel):
    """A client-supplied function available to the graph."""

    type: Literal["function"]
    name: Annotated[str, Field(min_length=1)]
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None
    # OpenAI v3 response objects include these fields during full tool replay.
    # Only the default synchronous shape fits LGOS's supported subset.
    allowed_callers: None = None
    async_: bool | None = Field(default=None, alias="async")
    defer_loading: None = None
    output_schema: None = None


class ResponseCustomTool(_ResponsesRequestModel):
    """Select one registered server tool with the Responses custom-tool shape."""

    type: Literal["custom"]
    name: Annotated[str, Field(min_length=1)]
    # Accept the default shape emitted by OpenAI v3 response objects while
    # continuing to reject unsupported custom-tool configuration.
    allowed_callers: None = None
    async_: bool | None = Field(default=None, alias="async")
    defer_loading: None = None
    description: None = None
    format: None = None


class ResponseWebSearchTool(_ResponsesRequestModel):
    """Select the graph's OpenAI-compatible web-search capability."""

    type: Literal["web_search"]


class ResponseNamedToolChoice(_ResponsesRequestModel):
    """Require one named function or custom tool."""

    type: Literal["function", "custom"]
    name: str


ResponseToolChoice: TypeAlias = (
    ResponseNamedToolChoice | Literal["none", "auto", "required"]
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

    # Background storage is accepted only by the polling path. Foreground
    # storage and conversation chaining remain outside the supported subset.
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
    "ResponseURLCitationInput",
    "ResponseWebSearchActionInput",
    "ResponseWebSearchCallInput",
    "ResponseWebSearchTool",
]
