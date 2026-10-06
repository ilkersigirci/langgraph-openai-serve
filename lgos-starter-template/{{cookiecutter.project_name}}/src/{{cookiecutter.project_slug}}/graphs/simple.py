"""A model call with typed, client-configurable runtime context."""

from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph_openai_serve import ClientSettings
from pydantic import BaseModel, Field

from {{ cookiecutter.project_slug }}.settings import (
    settings,
)


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
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[AgentState, SimpleContext, AgentState, AgentState]:
    chat_model = (
        model
        if model is not None
        else ChatOpenAI(
            model=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
        )
    )

    async def generate(
        state: AgentState, runtime: Runtime[SimpleContext]
    ) -> dict[str, list[BaseMessage]]:
        context = runtime.context or SimpleContext()
        history = state.messages if context.use_history else state.messages[-1:]
        response = await chat_model.ainvoke(
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
