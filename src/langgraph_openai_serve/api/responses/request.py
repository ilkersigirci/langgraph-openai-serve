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
from langgraph_openai_serve.core.errors import InvalidRequestError
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphConfig
from langgraph_openai_serve.graph.interrupt import InterruptResume
from langgraph_openai_serve.graph.request import (
    ClientFunctionTool,
    ClientToolChoice,
    GraphRequest,
    NamedCustomToolChoice,
    NamedFunctionToolChoice,
)


def decode_responses_request(
    request: ResponseCreateRequest,
    graph_config: GraphConfig,
) -> tuple[GraphRequest, list[BaseMessage], InterruptResume | None]:
    """Validate and normalize one stateless Responses request for its graph."""
    if request.background and not graph_config.supports(GraphFeature.BACKGROUND):
        message = f"Model '{request.model}' does not support background execution."
        raise InvalidRequestError(message, param="background")
    if request.previous_response_id is not None and not graph_config.supports(
        GraphFeature.INTERRUPTS
    ):
        message = (
            "Previous response state is not supported for model "
            f"'{request.model}'; only interruptible graphs support "
            "'previous_response_id'."
        )
        raise InvalidRequestError(message, param="previous_response_id")
    _validate_response_options(request)
    declarations = _tool_declarations(request, graph_config)
    choice = _decode_tool_choice(request.tool_choice, declarations)
    resume = parse_responses_resume(
        request.input,
        previous_response_id=request.previous_response_id,
    )
    graph_request = GraphRequest(
        model=request.model,
        metadata=dict(request.metadata or {}),
        user=request.safety_identifier or request.user,
        tools=tuple(
            ClientFunctionTool(
                name=tool.name,
                description=tool.description,
                parameters=dict(tool.parameters)
                if tool.parameters is not None
                else None,
                strict=tool.strict,
            )
            for tool in declarations.values()
            if isinstance(tool, ResponseFunctionTool)
        ),
        # tool_choice="none" leaves the graph's server tools unselected too.
        server_tools=()
        if choice == "none"
        else tuple(
            name for name, tool in declarations.items() if tool.type != "function"
        ),
        tool_choice=choice,
        parallel_tool_calls=request.parallel_tool_calls,
    )
    messages = (
        []
        if resume is not None
        else convert_responses_input(request.input, instructions=request.instructions)
    )
    return graph_request, messages, resume


def _decode_tool_choice(
    choice: ResponseToolChoice | None, declarations: dict[str, ResponseTool]
) -> ClientToolChoice | None:
    if choice == "required" and not declarations:
        msg = "tool_choice='required' needs at least one tool."
        raise InvalidRequestError(msg, param="tool_choice")
    if choice is None or isinstance(choice, str):
        return choice
    tool = declarations.get(choice.name)
    if tool is None or tool.type != choice.type:
        msg = "The named tool_choice must be declared in tools."
        raise InvalidRequestError(msg, param="tool_choice")
    if choice.type == "custom":
        return NamedCustomToolChoice(name=choice.name)
    return NamedFunctionToolChoice(name=choice.name)


def _tool_declarations(
    request: ResponseCreateRequest, graph_config: GraphConfig
) -> dict[str, ResponseTool]:
    """Index validated tools by the name used in graph calls and named choices."""
    declarations: dict[str, ResponseTool] = {}
    for index, tool in enumerate(request.tools or ()):
        name = tool.type if isinstance(tool, ResponseWebSearchTool) else tool.name
        if not isinstance(tool, ResponseWebSearchTool) and tool.async_:
            msg = (
                "Async tool calling ('async': true) is not supported for function "
                "or custom tools."
            )
            raise InvalidRequestError(msg, param=f"tools.{index}.async")
        if isinstance(tool, ResponseFunctionTool) and name in graph_config.server_tools:
            expected_type = "web_search" if name == "web_search" else "custom"
            msg = f"Registered server tool '{name}' must use type '{expected_type}'."
            raise InvalidRequestError(msg, param=f"tools.{index}.type")
        if isinstance(tool, ResponseCustomTool) and name == "web_search":
            msg = "The standard web_search tool must use type 'web_search'."
            raise InvalidRequestError(msg, param=f"tools.{index}.type")
        if (
            not isinstance(tool, ResponseFunctionTool)
            and name not in graph_config.server_tools
        ):
            msg = f"Tool '{name}' is not registered by model '{request.model}'."
            raise InvalidRequestError(
                msg,
                param=(
                    f"tools.{index}.name"
                    if isinstance(tool, ResponseCustomTool)
                    else f"tools.{index}.type"
                ),
            )
        if name in declarations:
            msg = f"Tool '{name}' is declared more than once."
            raise InvalidRequestError(msg, param="tools")
        declarations[name] = tool
    return declarations


def _validate_response_options(request: ResponseCreateRequest) -> None:
    if not request.background and request.store:
        message = "'store' must be false; response storage is not supported."
        raise InvalidRequestError(message, param="store")
    if request.conversation is not None:
        message = (
            "Responses conversations are not supported; resend the required input "
            "items."
        )
        raise InvalidRequestError(message, param="conversation")
    if request.previous_response_id is not None and request.instructions is not None:
        message = "'instructions' cannot be changed while resuming an interrupt."
        raise InvalidRequestError(message, param="instructions")
