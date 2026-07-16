"""Simple LLM-backed graph used by the demo API."""

from dataclasses import dataclass
from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel

from demo.api.settings import settings
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant called Langgraph Openai Serve. "
    "Chat with the user with a friendly tone."
)


class AgentState(BaseModel):
    """State passed through the simple message graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@dataclass(frozen=True)
class SimpleContext:
    """Immutable runtime context accepted by the demo graph."""

    use_history: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


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
    system_message = ("system", context.system_prompt)

    if context.use_history:
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


workflow = StateGraph(AgentState, context_schema=SimpleContext)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

simple_graph = workflow.compile()


def build_simple_context(
    request: ChatCompletionRequest,
    *,
    use_history: bool,
) -> SimpleContext:
    """Build simple graph context from OpenAI request metadata."""
    metadata = request.metadata or {}
    return SimpleContext(
        use_history=use_history,
        system_prompt=metadata.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
    )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SimpleContext",
    "build_simple_context",
    "simple_graph",
]
