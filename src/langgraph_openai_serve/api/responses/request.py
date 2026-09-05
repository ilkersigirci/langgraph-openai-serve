"""Decode Responses requests for protocol-neutral graph execution."""

from langchain_core.messages import BaseMessage

from langgraph_openai_serve.api.responses.interrupts import parse_responses_resume
from langgraph_openai_serve.api.responses.messages import convert_responses_input
from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
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
            ),
            tool_choice=_decode_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
        ),
        convert_responses_input(
            request.input,
            instructions=request.instructions,
        ),
        parse_responses_resume(request.input),
    )


def _decode_tool_choice(
    tool_choice: ResponseToolChoice | None,
) -> ClientToolChoice | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return NamedFunctionToolChoice(name=tool_choice.name)


def _validate_supported_semantics(request: ResponseCreateRequest) -> None:
    if request.store:
        message = "Response storage is not supported; 'store' must be false."
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
    if request.previous_response_id is not None:
        message = (
            "Previous response state is not supported; resend the required input items."
        )
        raise UnsupportedResponsesRequestError(
            message,
            param="previous_response_id",
        )


__all__ = ["UnsupportedResponsesRequestError", "decode_responses_request"]
