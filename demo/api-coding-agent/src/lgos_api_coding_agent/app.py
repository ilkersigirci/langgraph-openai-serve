"""Serve the coding agent through the ordinary LGOS /v1 contract."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph_openai_serve import (
    ClientSettings,
    GraphConfig,
    GraphRegistry,
    GraphRequest,
    LanggraphOpenaiServe,
)
from langgraph_openai_serve.protocol import CONVERSATION_METADATA_KEY
from pydantic import BaseModel

from lgos_api_coding_agent.codex_model import CodexChatModel
from lgos_api_coding_agent.codex_runtime import runtime_events
from lgos_api_coding_agent.settings import DESCRIPTION, GRAPH_ID, RuntimeSettings


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass(frozen=True)
class Conversation:
    """The agent thread a request continues; None runs it without memory."""

    thread_name: str | None


def conversation(
    request: GraphRequest, _settings: ClientSettings | None
) -> Conversation:
    conversation_id = request.metadata.get(CONVERSATION_METADATA_KEY)
    if not (request.user and conversation_id):
        return Conversation(thread_name=None)
    # Correlation values, not authorization: they only separate conversations.
    scope = f"{request.user}\0{conversation_id}"
    return Conversation(thread_name=sha256(scope.encode()).hexdigest())


def graph_config(model: BaseChatModel) -> GraphConfig:
    async def answer(
        state: State, runtime: Runtime[Conversation]
    ) -> dict[str, list[BaseMessage]]:
        reply = await model.ainvoke(
            state.messages, thread_name=runtime.context.thread_name
        )
        return {"messages": [reply]}

    graph = (
        StateGraph(State, context_schema=Conversation)
        .add_node("agent", answer)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile()
    )
    return GraphConfig(
        graph=graph, description=DESCRIPTION, context_factory=conversation
    )


def create_app(model: BaseChatModel | None = None) -> FastAPI:
    if model is None:
        settings = RuntimeSettings()
        model = CodexChatModel(
            event_source=runtime_events(settings), model_name=settings.model
        )
    registry = GraphRegistry(graphs={GRAPH_ID: graph_config(model)})
    return LanggraphOpenaiServe(registry=registry).bind_openai_api().app


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
