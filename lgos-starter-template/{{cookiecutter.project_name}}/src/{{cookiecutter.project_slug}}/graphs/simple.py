"""A model call with typed, client-configurable runtime context."""

from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph_openai_serve import ClientSettings
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


class SimpleContext(ClientSettings):
    use_history: bool = Field(
        default=True,
        title="Use conversation history",
        description="Include the history supplied by the client in this request.",
    )
    audience: Literal["general", "beginner", "expert"] = Field(
        default="general",
        title="Audience",
        description="Adapt explanations to the selected audience.",
    )


def create_simple_graph(
    model: BaseChatModel,
) -> CompiledStateGraph[AgentState, SimpleContext, AgentState, AgentState]:
    async def generate(
        state: AgentState, runtime: Runtime[SimpleContext]
    ) -> dict[str, list[BaseMessage]]:
        context = runtime.context or SimpleContext()
        history = state.messages if context.use_history else state.messages[-1:]
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=f"You are a helpful assistant. Explain for {context.audience} readers."
                ),
                *history,
            ]
        )
        return {"messages": [response]}

    workflow = StateGraph(AgentState, context_schema=SimpleContext)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
