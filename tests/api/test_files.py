"""Native file-input compatibility tests."""

import json
from http import HTTPStatus

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import AsyncOpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)


def _graph_app() -> FastAPI:
    def inspect_message(state: MessagesState) -> dict[str, list[AIMessage]]:
        content = state["messages"][-1].content
        return {"messages": [AIMessage(content=json.dumps(content))]}

    graph = (
        StateGraph(MessagesState)
        .add_node("inspect", inspect_message)
        .add_edge(START, "inspect")
        .add_edge("inspect", END)
        .compile()
    )
    registry = GraphRegistry(
        registry={"files": GraphConfig(graph=graph, description="Inspect inputs.")}
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


async def test_chat_file_id_reaches_graph_without_files_routes() -> None:
    app = _graph_app()
    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as http_client,
        AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=http_client,
            max_retries=0,
        ) as client,
    ):
        response = await client.chat.completions.create(
            model="files",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this file."},
                        {"type": "file", "file": {"file_id": "file-central"}},
                    ],
                }
            ],
        )
        files_response = await http_client.get("/v1/files")

    assert json.loads(response.choices[0].message.content or "null") == [
        {"type": "text", "text": "Read this file."},
        {"type": "file", "file": {"file_id": "file-central"}},
    ]
    assert files_response.status_code == HTTPStatus.NOT_FOUND
