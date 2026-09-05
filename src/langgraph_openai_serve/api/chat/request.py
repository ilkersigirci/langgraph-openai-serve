"""Decode Chat Completions requests for protocol-neutral graph execution."""

from langchain_core.messages import BaseMessage

from langgraph_openai_serve.api.chat.messages import convert_to_lc_messages
from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatToolChoice,
)
from langgraph_openai_serve.api.chat.utils.interrupts import parse_resume_request
from langgraph_openai_serve.graph.interrupt.models import InterruptResume
from langgraph_openai_serve.graph.request import (
    ClientFunctionTool,
    ClientToolChoice,
    GraphRequest,
    NamedFunctionToolChoice,
)


def decode_chat_request(
    request: ChatCompletionRequest,
) -> tuple[GraphRequest, list[BaseMessage], InterruptResume | None]:
    """Normalize one Chat Completions request for graph execution."""
    resume = parse_resume_request(request.messages)
    graph_request = GraphRequest(
        model=request.model,
        metadata=dict(request.metadata or {}),
        user=request.user,
        tools=tuple(
            ClientFunctionTool(
                name=tool.function.name,
                description=tool.function.description,
                parameters=(
                    dict(tool.function.parameters)
                    if tool.function.parameters is not None
                    else None
                ),
                strict=tool.function.strict,
            )
            for tool in request.tools or ()
        ),
        tool_choice=_decode_tool_choice(request.tool_choice),
        parallel_tool_calls=request.parallel_tool_calls,
    )
    return (
        graph_request,
        convert_to_lc_messages(request.messages),
        resume,
    )


def _decode_tool_choice(
    tool_choice: ChatToolChoice | None,
) -> ClientToolChoice | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return NamedFunctionToolChoice(name=tool_choice.function.name)


__all__ = ["decode_chat_request"]
