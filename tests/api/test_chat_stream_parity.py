import pytest
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from openai import APIError, AsyncOpenAI

from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from tests.graph.support.schemas import MessageState


def deterministic_app() -> FastAPI:
    async def generate(_state: MessageState):
        return {"messages": [AIMessage(content="deterministic answer")]}

    fallback_graph = (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )

    model = FakeListChatModel(responses=["live text"])

    async def generate_mismatch(state: MessageState):
        await model.ainvoke(state["messages"])
        return {"messages": [AIMessage(content="durable text")]}

    mismatch_graph = (
        StateGraph(MessageState)
        .add_node("generate", generate_mismatch)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )
    registry = GraphRegistry(
        registry={
            "deterministic": GraphConfig(
                graph=fallback_graph,
                description="DUMMY",
            ),
            "mismatch": GraphConfig(
                graph=mismatch_graph,
                description="DUMMY",
                streamable_node_names=["generate"],
            ),
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


async def test_streaming_fails_when_text_differs_from_the_final_message(
    openai_client: AsyncOpenAI,
) -> None:
    complete = await openai_client.chat.completions.create(
        model="mismatch",
        messages=[{"role": "user", "content": "Hi"}],
    )
    async with openai_client.chat.completions.stream(
        model="mismatch",
        messages=[{"role": "user", "content": "Hi"}],
    ) as stream:
        with pytest.raises(APIError, match="Internal server error"):
            await stream.until_done()
        streamed = stream.current_completion_snapshot.choices[0]

    assert complete.choices[0].message.content == "durable text"
    assert streamed.message.content == "live text"
    assert streamed.finish_reason is None
