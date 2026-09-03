"""Simple model graph with tools supplied by the OpenAI client."""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest, Tool
from pydantic import BaseModel, Field

from lgos_demo_api.graphs.simple import DEFAULT_SYSTEM_PROMPT
from lgos_demo_api.settings import settings

ToolChoice = str | bool | dict[str, object]


class ExternalToolsState(BaseModel):
    """Messages and client-owned tools for one model invocation."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None


def request_to_input(
    request: ChatCompletionRequest,
    messages: list[BaseMessage],
) -> ExternalToolsState:
    """Keep client-provided tools with the messages sent to the model."""
    return ExternalToolsState(
        messages=messages,
        tools=request.tools or [],
        tool_choice=request.tool_choice,
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
        tools = [
            tool.model_dump(mode="json", exclude_none=True) for tool in state.tools
        ]
        if state.tool_choice is None:
            model_response = await model.bind_tools(tools).ainvoke(conversation)
        else:
            model_response = await model.bind_tools(
                tools,
                tool_choice=state.tool_choice,
            ).ainvoke(conversation)
    else:
        model_response = await model.ainvoke(conversation)

    return {"messages": [model_response]}


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
