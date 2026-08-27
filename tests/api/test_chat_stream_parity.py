import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from openai import AsyncOpenAI

from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from tests.graph.support.schemas import MessageState


def deterministic_app() -> FastAPI:
    async def generate(_state: MessageState):
        return {"messages": [AIMessage(content="deterministic answer")]}

    graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    registry = GraphRegistry(
        registry={
            "deterministic": GraphConfig(
                graph=graph,
                description="DUMMY",
            )
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


@pytest.fixture
def fastapi_app() -> FastAPI:
    return deterministic_app()


async def test_streaming_falls_back_to_the_final_message(
    openai_client: AsyncOpenAI,
) -> None:
    complete = await openai_client.chat.completions.create(
        model="deterministic",
        messages=[{"role": "user", "content": "Hi"}],
    )
    stream = await openai_client.chat.completions.create(
        model="deterministic",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    streamed = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert streamed == complete.choices[0].message.content == "deterministic answer"
