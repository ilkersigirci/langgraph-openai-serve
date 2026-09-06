"""Decode Responses requests for protocol-neutral graph execution."""

from langchain_core.messages import BaseMessage

from langgraph_openai_serve.api.responses.interrupts import parse_responses_resume
from langgraph_openai_serve.api.responses.messages import convert_responses_input
from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
    ResponseFunctionTool,
    ResponseHostedTool,
    ResponseToolChoice,
)
from langgraph_openai_serve.graph.interrupt.models import InterruptResume
from langgraph_openai_serve.graph.request import (
    ClientFunctionTool,
    ClientToolChoice,
    GraphRequest,
    NamedFunctionToolChoice,
)


class UnsupportedResponsesRequestError(ValueError):
    """Raised when a valid OpenAI field has unsupported LGOS semantics."""

    def __init__(self, message: str, *, param: str) -> None:
        super().__init__(message)
        self.param = param


def decode_responses_request(
    request: ResponseCreateRequest,
) -> tuple[GraphRequest, list[BaseMessage], InterruptResume | None]:
    """Normalize one supported, stateless Responses request."""
    _validate_supported_semantics(request)
    resume = parse_responses_resume(
        request.input,
        previous_response_id=request.previous_response_id,
    )
    return (
        GraphRequest(
            model=request.model,
            metadata=dict(request.metadata or {}),
            user=request.user,
            tools=tuple(
                ClientFunctionTool(
                    name=tool.name,
                    description=tool.description,
                    parameters=(
                        dict(tool.parameters) if tool.parameters is not None else None
                    ),
                    strict=tool.strict,
                )
                for tool in request.tools or ()
                if isinstance(tool, ResponseFunctionTool)
            ),
            hosted_tools=tuple(
                tool.name
                for tool in request.tools or ()
                if isinstance(tool, ResponseHostedTool)
            ),
            tool_choice=_decode_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
        ),
        (
            []
            if resume is not None
            else convert_responses_input(
                request.input,
                instructions=request.instructions,
            )
        ),
        resume,
    )


def validate_hosted_tools(request: ResponseCreateRequest, supported: set[str]) -> None:
    """Reject unavailable hosted tools before graph execution or SSE starts."""
    for index, tool in enumerate(request.tools or ()):
        if isinstance(tool, ResponseHostedTool) and tool.name not in supported:
            message = f"Hosted tool '{tool.name}' is not supported by model '{request.model}'."
            raise UnsupportedResponsesRequestError(message, param=f"tools.{index}.name")


def _decode_tool_choice(
    tool_choice: ResponseToolChoice | None,
) -> ClientToolChoice | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return NamedFunctionToolChoice(name=tool_choice.name)


def _validate_supported_semantics(request: ResponseCreateRequest) -> None:
    if request.store:
        message = "'store' must be false; response storage is not supported."
        raise UnsupportedResponsesRequestError(message, param="store")
    if request.background:
        message = "Background Responses are not supported."
        raise UnsupportedResponsesRequestError(message, param="background")
    if request.conversation is not None:
        message = (
            "Responses conversations are not supported; resend the required input "
            "items."
        )
        raise UnsupportedResponsesRequestError(message, param="conversation")
    if request.previous_response_id is not None and request.instructions is not None:
        message = "'instructions' cannot be changed while resuming an interrupt."
        raise UnsupportedResponsesRequestError(message, param="instructions")


__all__ = [
    "UnsupportedResponsesRequestError",
    "decode_responses_request",
    "validate_hosted_tools",
]
