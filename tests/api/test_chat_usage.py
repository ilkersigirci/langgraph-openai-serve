import json

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from openai import AsyncOpenAI
from openai.types import CompletionUsage

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LanggraphOpenaiServe,
)
from tests.graph.support.schemas import MessageState

USAGE = {
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5,
}


def usage_graph() -> object:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="hello",
                response_metadata={"model_name": "usage-model"},
                usage_metadata={
                    "input_tokens": USAGE["prompt_tokens"],
                    "output_tokens": USAGE["completion_tokens"],
                    "total_tokens": USAGE["total_tokens"],
                },
            )
        ]
    )

    async def generate(state: MessageState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    return (
        StateGraph(MessageState)
        .add_node("generate", generate)
        .set_entry_point("generate")
        .set_finish_point("generate")
        .compile()
    )


def usage_app() -> FastAPI:
    registry = GraphRegistry(
        registry={
            "usage": GraphConfig(
                graph=usage_graph,
                description="DUMMY",
                streamable_node_names=["generate"],
            )
        }
    )
    return LanggraphOpenaiServe(graphs=registry).bind_openai_api().app


@pytest.fixture
def fastapi_app() -> FastAPI:
    return usage_app()


def _assert_usage(usage: CompletionUsage | None) -> None:
    assert usage is not None
    assert usage.prompt_tokens == USAGE["prompt_tokens"]
    assert usage.completion_tokens == USAGE["completion_tokens"]
    assert usage.total_tokens == USAGE["total_tokens"]


async def test_non_streaming_uses_provider_reported_usage(client) -> None:
    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=client,
        max_retries=0,
    ) as openai_client:
        response = await openai_client.chat.completions.create(
            model="usage",
            messages=[{"role": "user", "content": "Hi"}],
        )

    _assert_usage(response.usage)


async def test_streaming_usage_uses_the_standard_final_chunk(client) -> None:
    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=client,
        max_retries=0,
    ) as openai_client:
        stream = await openai_client.chat.completions.create(
            model="usage",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]

    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].choices == []
    _assert_usage(usage_chunks[0].usage)


async def test_streaming_usage_is_null_on_ordinary_wire_chunks(
    client: AsyncClient,
) -> None:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "usage",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        payloads = [
            json.loads(line.removeprefix("data: "))
            async for line in response.aiter_lines()
            if line and line != "data: [DONE]"
        ]

    ordinary_chunks = [payload for payload in payloads if payload["choices"]]
    assert ordinary_chunks
    assert all(payload.get("usage") is None for payload in ordinary_chunks)
    assert all("usage" in payload for payload in ordinary_chunks)


async def test_streaming_omits_usage_unless_requested(client) -> None:
    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=client,
        max_retries=0,
    ) as openai_client:
        stream = await openai_client.chat.completions.create(
            model="usage",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

    assert all(chunk.usage is None for chunk in chunks)
