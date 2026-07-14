import json
from collections.abc import AsyncIterator
from http import HTTPStatus

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, BadRequestError

from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from tests.graph.support.interrupt import make_interrupt_graph

pytestmark = pytest.mark.anyio

MODEL = "interruptible"
THREAD_ID = "thread-1"
INTERRUPT_PAYLOAD = {"question": "Approve?"}


async def _create_completion(openai_client: AsyncOpenAI, *, stream: bool = False):
    return await openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        stream=stream,
        metadata={"langgraph_thread_id": THREAD_ID},
    )


def _assert_interrupt_tool_call(tool_call) -> None:
    assert tool_call.function is not None
    assert tool_call.function.name == "langgraph_interrupt"
    assert tool_call.id is not None
    assert tool_call.id.startswith("lg_interrupt_")
    arguments = json.loads(tool_call.function.arguments)
    assert arguments == {
        "version": 1,
        "kind": "hitl.interrupt",
        "thread_id": THREAD_ID,
        "interrupt_id": tool_call.id.removeprefix("lg_interrupt_"),
        "payload": INTERRUPT_PAYLOAD,
    }


async def _resume_interrupt(
    openai_client: AsyncOpenAI,
    response,
    resume_value: str,
):
    tool_call = response.choices[0].message.tool_calls[0]
    return await openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"resume": resume_value}),
            }
        ],
        metadata={"langgraph_thread_id": THREAD_ID},
    )


@pytest.fixture
async def fastapi_app() -> AsyncIterator[FastAPI]:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph_registry = GraphRegistry(
            registry={
                MODEL: GraphConfig(
                    graph=make_interrupt_graph(
                        INTERRUPT_PAYLOAD,
                        checkpointer=checkpointer,
                    ),
                    interrupts_enabled=True,
                ),
            }
        )
        yield LanggraphOpenaiServe(graphs=graph_registry).bind_openai_api().app


async def test_non_streaming_interrupt_matches_openai_tool_call_contract(
    openai_client: AsyncOpenAI,
) -> None:
    response = await _create_completion(openai_client)

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    _assert_interrupt_tool_call(choice.message.tool_calls[0])


async def test_streaming_interrupt_missing_thread_id_returns_400_before_sse(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

    response = exc_info.value.response
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert response.json() == {
        "error": {
            "message": (
                "metadata.langgraph_thread_id is required for interrupt-enabled graphs."
            ),
            "type": "invalid_request_error",
            "param": "metadata.langgraph_thread_id",
            "code": None,
        }
    }


async def test_streaming_interrupt_matches_openai_tool_call_contract(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await _create_completion(openai_client, stream=True)
    chunks = [chunk async for chunk in stream]

    tool_call_chunks = [
        chunk for chunk in chunks if chunk.choices[0].delta.tool_calls is not None
    ]
    assert len(tool_call_chunks) == 1
    tool_call = tool_call_chunks[0].choices[0].delta.tool_calls[0]
    assert tool_call.index == 0
    _assert_interrupt_tool_call(tool_call)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


async def test_tool_response_resumes_interrupted_completion(
    openai_client: AsyncOpenAI,
) -> None:
    first_response = await _create_completion(openai_client)
    final_response = await _resume_interrupt(
        openai_client,
        first_response,
        "approve",
    )

    assert final_response.choices[0].message.content == "resumed:approve"
