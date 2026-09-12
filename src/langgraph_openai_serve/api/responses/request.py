"""Decode Responses requests for protocol-neutral graph execution."""

from langchain_core.messages import BaseMessage

from langgraph_openai_serve.api.responses.interrupts import parse_responses_resume
from langgraph_openai_serve.api.responses.messages import convert_responses_input
from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
    ResponseCustomTool,
    ResponseFunctionTool,
    ResponseTool,
    ResponseToolChoice,
    ResponseWebSearchTool,
)
from langgraph_openai_serve.graph.interrupt.models import InterruptResume
from langgraph_openai_serve.graph.request import (
    ClientFunctionTool,
    ClientToolChoice,
    GraphRequest,
    NamedCustomToolChoice,
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
            hosted_tools=selected_hosted_tools(request),
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


def _decode_tool_choice(
    tool_choice: ResponseToolChoice | None,
) -> ClientToolChoice | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return (
        NamedCustomToolChoice(name=tool_choice.name)
        if tool_choice.type == "custom"
        else NamedFunctionToolChoice(name=tool_choice.name)
    )


def selected_hosted_tools(request: ResponseCreateRequest) -> tuple[str, ...]:
    """Return the registered server tools selected for this response."""
    if request.tool_choice == "none":
        return ()
    return tuple(
        _tool_name(tool)
        for tool in request.tools or ()
        if isinstance(tool, (ResponseCustomTool, ResponseWebSearchTool))
    )


def validate_tools(request: ResponseCreateRequest, hosted_tools: set[str]) -> None:
    """Reject unknown server-tool selectors before execution or SSE starts."""
    declarations: dict[str, str] = {}
    for index, tool in enumerate(request.tools or ()):
        name = _tool_name(tool)
        if isinstance(tool, ResponseCustomTool) and name == "web_search":
            msg = "The standard web_search tool must use type 'web_search'."
            raise UnsupportedResponsesRequestError(msg, param=f"tools.{index}.type")
        if isinstance(tool, (ResponseCustomTool, ResponseWebSearchTool)) and (
            name not in hosted_tools
        ):
            msg = f"Tool '{name}' is not hosted by model '{request.model}'."
            raise UnsupportedResponsesRequestError(
                msg,
                param=(
                    f"tools.{index}.name"
                    if isinstance(tool, ResponseCustomTool)
                    else f"tools.{index}.type"
                ),
            )
        if name in declarations:
            msg = f"Tool '{name}' is declared more than once."
            raise UnsupportedResponsesRequestError(msg, param="tools")
        declarations[name] = tool.type
    choice = request.tool_choice
    if choice is not None and not isinstance(choice, str):
        if declarations.get(choice.name) != choice.type:
            msg = "The named tool_choice must be declared in tools."
            raise UnsupportedResponsesRequestError(msg, param="tool_choice")
    elif choice == "required" and not declarations:
        msg = "tool_choice='required' needs at least one tool."
        raise UnsupportedResponsesRequestError(msg, param="tool_choice")


def _tool_name(tool: ResponseTool) -> str:
    return tool.type if isinstance(tool, ResponseWebSearchTool) else tool.name


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
    "selected_hosted_tools",
    "validate_tools",
]
