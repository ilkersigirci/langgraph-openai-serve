"""Shared state, conversion, and model invocation for client-owned tools."""

from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph_openai_serve import (
    ClientFunctionTool,
    ClientToolChoice,
    GraphRequest,
    NamedFunctionToolChoice,
)
from pydantic import BaseModel, Field

from lgos_demo_api.utils.models import chat_completions_model


class ClientToolsState(BaseModel):
    """Messages and client-owned tools for one model invocation."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools: tuple[ClientFunctionTool, ...] = Field(default_factory=tuple)
    tool_choice: ClientToolChoice | None = None
    parallel_tool_calls: bool | None = None


def request_to_input(
    request: GraphRequest,
    messages: list[BaseMessage],
) -> ClientToolsState:
    """Keep client-provided tools with the messages sent to the model."""
    return ClientToolsState(
        messages=messages,
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
    )


async def invoke_client_tool_model(
    state: ClientToolsState,
    *,
    system_prompt: str,
    default_tool_choice: ClientToolChoice | None = None,
    model: ChatOpenAI | None = None,
) -> AIMessage:
    """Invoke a chat model, by default the shared one, leaving tools to the client."""
    model = model or chat_completions_model()
    conversation = [SystemMessage(content=system_prompt), *state.messages]
    if not state.tools:
        return await model.ainvoke(conversation)
    tool_choice = state.tool_choice or default_tool_choice
    # ChatOpenAI leaves out a None tool choice or parallel_tool_calls.
    return await model.bind_tools(
        [chat_tool(tool) for tool in state.tools],
        tool_choice=_chat_tool_choice(tool_choice) if tool_choice else None,
        parallel_tool_calls=state.parallel_tool_calls,
    ).ainvoke(conversation)


def chat_tool(tool: ClientFunctionTool) -> dict[str, object]:
    """Convert a client-owned function to LangChain's OpenAI tool format."""
    function: dict[str, object] = {"name": tool.name}
    if tool.description is not None:
        function["description"] = tool.description
    if tool.parameters is not None:
        function["parameters"] = dict(tool.parameters)
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _chat_tool_choice(tool_choice: ClientToolChoice) -> str | dict[str, object]:
    if isinstance(tool_choice, NamedFunctionToolChoice):
        return {
            "type": "function",
            "function": {"name": tool_choice.name},
        }
    if isinstance(tool_choice, str):
        return tool_choice
    msg = "This graph only supports client-owned function tools."
    raise ValueError(msg)
