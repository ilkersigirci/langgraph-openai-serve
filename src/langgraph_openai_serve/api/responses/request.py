"""Decode Responses requests for protocol-neutral graph execution."""

from collections.abc import Set as AbstractSet

from langchain_core.messages import BaseMessage

from langgraph_openai_serve.api.responses.interrupts import parse_responses_resume
from langgraph_openai_serve.api.responses.messages import convert_responses_input
from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
    ResponseCustomTool,
    ResponseCustomToolCallInput,
    ResponseFunctionCallInput,
    ResponseFunctionTool,
    ResponseTool,
    ResponseToolChoice,
    ResponseWebSearchTool,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig
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
    server_tools: AbstractSet[str],
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
            server_tools=selected_server_tools(request, server_tools),
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


def decode_graph_request(
    request: ResponseCreateRequest,
    graph_config: GraphConfig,
) -> tuple[GraphRequest, list[BaseMessage], InterruptResume | None]:
    """Validate and normalize one Responses request for its registered graph."""
    _validate_tools(request, graph_config.server_tools)
    if request.previous_response_id is not None and not graph_config.supports(
        GraphFeature.INTERRUPTS
    ):
        message = (
            "Previous response state is not supported for model "
            f"'{request.model}'; only interruptible graphs support "
            "'previous_response_id'."
        )
        raise UnsupportedResponsesRequestError(message, param="previous_response_id")
    return decode_responses_request(request, graph_config.server_tools)


def _decode_tool_choice(
    tool_choice: ResponseToolChoice | None,
) -> ClientToolChoice | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if tool_choice.type == "custom":
        return NamedCustomToolChoice(name=tool_choice.name)
    return NamedFunctionToolChoice(name=tool_choice.name)


def selected_server_tools(
    request: ResponseCreateRequest, server_tools: AbstractSet[str]
) -> tuple[str, ...]:
    """Return the registered server tools selected for this response."""
    if request.tool_choice == "none":
        return ()
    return tuple(
        _tool_name(tool)
        for tool in request.tools or ()
        if isinstance(tool, (ResponseCustomTool, ResponseWebSearchTool))
        and _tool_name(tool) in server_tools
    )


def _validate_tools(
    request: ResponseCreateRequest,
    server_tools: AbstractSet[str],
) -> None:
    """Reject unknown server-tool selectors before execution or SSE starts."""
    declarations: dict[str, str] = {}
    for index, tool in enumerate(request.tools or ()):
        name = _tool_name(tool)
        if isinstance(tool, ResponseFunctionTool) and name in server_tools:
            expected_type = "web_search" if name == "web_search" else "custom"
            msg = f"Registered server tool '{name}' must use type '{expected_type}'."
            raise UnsupportedResponsesRequestError(msg, param=f"tools.{index}.type")
        if isinstance(tool, ResponseCustomTool) and name == "web_search":
            msg = "The standard web_search tool must use type 'web_search'."
            raise UnsupportedResponsesRequestError(msg, param=f"tools.{index}.type")
        if isinstance(tool, (ResponseCustomTool, ResponseWebSearchTool)) and (
            name not in server_tools
        ):
            msg = f"Tool '{name}' is not registered by model '{request.model}'."
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
    _validate_storage_mode(request)
    _validate_tool_replay_mode(request)
    if request.conversation is not None:
        message = (
            "Responses conversations are not supported; resend the required input "
            "items."
        )
        raise UnsupportedResponsesRequestError(message, param="conversation")
    if request.previous_response_id is not None and request.instructions is not None:
        message = "'instructions' cannot be changed while resuming an interrupt."
        raise UnsupportedResponsesRequestError(message, param="instructions")


def _validate_storage_mode(request: ResponseCreateRequest) -> None:
    if not request.background:
        if request.store:
            message = "'store' must be false; response storage is not supported."
            raise UnsupportedResponsesRequestError(message, param="store")
        return
    if request.stream:
        message = (
            "Background responses do not support streaming. Set stream=false or "
            "omit it, then retrieve the response by ID."
        )
        raise UnsupportedResponsesRequestError(message, param="stream")


def _validate_tool_replay_mode(request: ResponseCreateRequest) -> None:
    for index, tool in enumerate(request.tools or ()):
        if isinstance(tool, (ResponseFunctionTool, ResponseCustomTool)) and (
            tool.async_ is True
        ):
            message = (
                "Async tool calling ('async': true) is not supported for function "
                "or custom tools."
            )
            raise UnsupportedResponsesRequestError(
                message,
                param=f"tools.{index}.async",
            )
    if isinstance(request.input, list):
        for index, item in enumerate(request.input):
            if (
                isinstance(
                    item,
                    (ResponseFunctionCallInput, ResponseCustomToolCallInput),
                )
                and item.async_ is True
            ):
                message = "Asynchronous tool-call replay is not supported."
                raise UnsupportedResponsesRequestError(
                    message,
                    param=f"input.{index}.async",
                )


__all__ = [
    "UnsupportedResponsesRequestError",
    "decode_graph_request",
    "decode_responses_request",
    "selected_server_tools",
]
