"""Simple LLM-backed graph used by the demo API."""

from collections.abc import Sequence
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph_openai_serve import ClientSettings, GraphConfig
from pydantic import AfterValidator, BaseModel, Field

from lgos_demo_api.core.settings import settings
from lgos_demo_api.utils.models import chat_completions_model

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant called Langgraph Openai Serve. "
    "Chat with the user with a friendly tone."
)
# Both configured models accept plain Chat Completions requests without tools.
MODEL_CHOICES = tuple(
    dict.fromkeys(
        (settings.OPENAI_CHAT_COMPLETIONS_MODEL, settings.OPENAI_RESPONSES_MODEL)
    )
)


def _offered_model(model: str) -> str:
    if model not in MODEL_CHOICES:
        msg = f"Choose one of: {', '.join(MODEL_CHOICES)}"
        raise ValueError(msg)
    return model


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
    audience: Literal["general", "beginner", "expert"] = Field(
        default="general",
        title="Audience",
        description="Adapt terminology and assumed knowledge to the selected audience.",
    )
    model: Annotated[str, AfterValidator(_offered_model)] = Field(
        default=MODEL_CHOICES[0],
        title="Model",
        description="Gateway model that writes the answer.",
        json_schema_extra={"enum": list(MODEL_CHOICES)},
    )


async def generate(
    state: AgentState,
    runtime: Runtime[SimpleContext],
) -> dict[str, list[AIMessage]]:
    """Generate a response to the latest message in the graph state."""
    context = runtime.context or SimpleContext()
    model = chat_completions_model(context.model)
    messages = state.messages if context.use_history else state.messages[-1:]
    conversation = [
        SystemMessage(
            content=(
                f"{DEFAULT_SYSTEM_PROMPT} "
                f"Adapt explanations for {context.audience} readers."
            )
        ),
        *messages,
    ]

    response = await model.ainvoke(conversation)
    return {"messages": [response]}


workflow = StateGraph(AgentState, context_schema=SimpleContext)
workflow.add_node("generate", generate)
workflow.add_edge("generate", END)
workflow.set_entry_point("generate")

simple_graph = workflow.compile()

simple_graph_config = GraphConfig(
    graph=simple_graph,
    description=(
        "Streams model output with configurable history and audience settings."
    ),
    client_settings=SimpleContext,
)

__all__ = ["simple_graph", "simple_graph_config"]
