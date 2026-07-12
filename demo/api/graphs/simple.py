"""Simple LLM-backed graph used by the demo API."""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from demo.api.settings import settings


class AgentState(BaseModel):
    """State passed through the simple message graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class SimpleConfigSchema(BaseModel):
    """Configurable fields accepted by the demo graph."""

    use_history: bool = False


async def generate(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict[str, list[AIMessage]]:
    """Generate a response to the latest message in the graph state."""
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.7,
        streaming=True,
    )
    system_message = (
        "system",
        "You are a helpful assistant called Langgraph Openai Serve. "
        "Chat with the user with friendly tone",
    )
    configurable = SimpleConfigSchema.model_validate(
        (config or {}).get("configurable", {})
    )

    if configurable.use_history:
        prompt = ChatPromptTemplate.from_messages([system_message, *state.messages])
        response = await (prompt | model | StrOutputParser()).ainvoke({})
    else:
        prompt = ChatPromptTemplate.from_messages(
            [system_message, ("human", "{question}")]
        )
        response = await (prompt | model | StrOutputParser()).ainvoke(
            {"question": state.messages[-1].content}
        )

    return {"messages": [AIMessage(content=response)]}


workflow = StateGraph(AgentState, context_schema=SimpleConfigSchema)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

simple_graph = workflow.compile()

__all__ = ["simple_graph"]
