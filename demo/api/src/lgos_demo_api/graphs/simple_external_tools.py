"""Simple model graph with tools supplied by the OpenAI client."""

from typing import Annotated, Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import (
    ClientFunctionTool,
    ClientToolChoice,
    GraphConfig,
    GraphRequest,
    NamedFunctionToolChoice,
)
from pydantic import BaseModel, Field

from lgos_demo_api.graphs.simple import DEFAULT_SYSTEM_PROMPT
from lgos_demo_api.settings import settings


class ExternalToolsState(BaseModel):
    """Messages and client-owned tools for one model invocation."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools: tuple[ClientFunctionTool, ...] = Field(default_factory=tuple)
    tool_choice: ClientToolChoice | None = None
    parallel_tool_calls: bool | None = None


def request_to_input(
    request: GraphRequest,
    messages: list[BaseMessage],
) -> ExternalToolsState:
    """Keep client-provided tools with the messages sent to the model."""
    return ExternalToolsState(
        messages=messages,
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
    )


async def generate(state: ExternalToolsState) -> dict[str, list[AIMessage]]:
    """Return a model response without executing client-owned tools."""
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.7,
        streaming=True,
    )
    conversation = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT), *state.messages]

    if state.tools:
        binding_options: dict[str, Any] = {}
        if state.tool_choice is not None:
            binding_options["tool_choice"] = _chat_tool_choice(state.tool_choice)
        if state.parallel_tool_calls is not None:
            binding_options["parallel_tool_calls"] = state.parallel_tool_calls
        model_response = await model.bind_tools(
            [_chat_tool(tool) for tool in state.tools],
            **binding_options,
        ).ainvoke(conversation)
    else:
        model_response = await model.ainvoke(conversation)

    return {"messages": [model_response]}


def _chat_tool(tool: ClientFunctionTool) -> dict[str, object]:
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
    return tool_choice


workflow = StateGraph(ExternalToolsState)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

simple_external_tools_graph = workflow.compile()

simple_external_tools_graph_config = GraphConfig(
    graph=simple_external_tools_graph,
    description=(
        "Streams a chat model response with tools supplied and executed by the client."
    ),
    streamable_node_names=["generate"],
    request_to_input=request_to_input,
)

__all__ = [
    "ExternalToolsState",
    "request_to_input",
    "simple_external_tools_graph",
    "simple_external_tools_graph_config",
]
