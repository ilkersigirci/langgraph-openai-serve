"""Simple LLM-backed graph used by the demo API."""

from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from demo.api.settings import settings
from langgraph_openai_serve import ClientSettings

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant called Langgraph Openai Serve. "
    "Chat with the user with a friendly tone."
)


class AgentState(BaseModel):
    """State passed through the simple message graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class SimpleContext(ClientSettings):
    """Allowlisted runtime context configurable by ordinary OpenAI clients."""

    use_history: bool = Field(
        default=False,
        title="Use conversation history",
        description="Include prior user and assistant messages in each generation.",
    )


async def generate(
    state: AgentState,
    runtime: Runtime[SimpleContext],
) -> dict[str, list[AIMessage]]:
    """Generate a response to the latest message in the graph state."""
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.7,
        streaming=True,
    )
    context = runtime.context or SimpleContext()
    messages = state.messages if context.use_history else state.messages[-1:]
    conversation = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
        *messages,
    ]

    response = await (model | StrOutputParser()).ainvoke(conversation)

    return {"messages": [AIMessage(content=response)]}


workflow = StateGraph(AgentState, context_schema=SimpleContext)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

simple_graph = workflow.compile()
