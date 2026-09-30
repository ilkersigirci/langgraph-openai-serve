"""Serve the coding agent through the ordinary LGOS /v1 contract."""

from typing import Annotated

import uvicorn
from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from pydantic import BaseModel

from lgos_api_coding_agent.codex_model import CodexChatModel
from lgos_api_coding_agent.codex_runtime import runtime_events
from lgos_api_coding_agent.settings import DESCRIPTION, GRAPH_ID, RuntimeSettings


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


def graph_config(model: BaseChatModel) -> GraphConfig:
    async def answer(state: State) -> dict[str, list[BaseMessage]]:
        return {"messages": [await model.ainvoke(state.messages)]}

    graph = (
        StateGraph(State)
        .add_node("agent", answer)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile()
    )
    return GraphConfig(graph=graph, description=DESCRIPTION)


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
