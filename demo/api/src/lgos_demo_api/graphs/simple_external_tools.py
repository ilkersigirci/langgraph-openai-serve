"""Simple model graph with tools supplied by the OpenAI client."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph_openai_serve import GraphConfig

from lgos_demo_api.graphs.simple import DEFAULT_SYSTEM_PROMPT
from lgos_demo_api.utils.client_tools import (
    ClientToolsState,
    invoke_client_tool_model,
    request_to_input,
)


async def generate(state: ClientToolsState) -> dict[str, list[AIMessage]]:
    """Return a model response without executing client-owned tools."""
    model_response = await invoke_client_tool_model(
        state,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=0.7,
    )
    return {"messages": [model_response]}


workflow = StateGraph(ClientToolsState)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

simple_external_tools_graph = workflow.compile()

simple_external_tools_graph_config = GraphConfig(
    graph=simple_external_tools_graph,
    description=(
        "Streams a chat model response with tools supplied and executed by the client."
    ),
    request_to_input=request_to_input,
)

__all__ = ["simple_external_tools_graph", "simple_external_tools_graph_config"]
