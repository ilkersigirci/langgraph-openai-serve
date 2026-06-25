import json
from http import HTTPStatus

import pytest
from openai import BadRequestError, OpenAI

from langgraph_openai_serve import GraphConfig, GraphRegistry
from tests.graph.support.interrupt import make_interrupt_graph

MODEL = "interruptible"
THREAD_ID = "thread-1"
INTERRUPT_PAYLOAD = {"question": "Approve?"}


def create_completion(openai_client: OpenAI, *, stream: bool = False):
    return openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        stream=stream,
        metadata={"langgraph_thread_id": THREAD_ID},
    )


def assert_interrupt_tool_call(tool_call) -> None:
    assert tool_call.function is not None
    assert tool_call.function.name == "langgraph_interrupt"
    assert_interrupt_arguments(json.loads(tool_call.function.arguments))


def assert_interrupt_arguments(arguments: dict) -> None:
    assert arguments["version"] == 1
    assert arguments["kind"] == "hitl.interrupt"
    assert arguments["thread_id"] == THREAD_ID
    assert arguments["payload"] == INTERRUPT_PAYLOAD


def interrupt_payload(response) -> dict:
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    return arguments["payload"]


def resume_interrupt(openai_client: OpenAI, response, resume_value: str):
    tool_call = response.choices[0].message.tool_calls[0]
    return openai_client.chat.completions.create(
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
def graph_registry() -> GraphRegistry:
    return GraphRegistry(
        registry={
            MODEL: GraphConfig(
                graph=make_interrupt_graph(INTERRUPT_PAYLOAD),
                interrupts_enabled=True,
            ),
        }
    )


def test_non_streaming_interrupt_returns_tool_calls(
    openai_client: OpenAI,
) -> None:
    response = create_completion(openai_client)

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert_interrupt_tool_call(choice.message.tool_calls[0])


def test_streaming_interrupt_missing_thread_id_returns_400_before_sse(
    openai_client: OpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        openai_client.chat.completions.create(
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
                "metadata.langgraph_thread_id is required for "
                "interrupt-enabled graphs."
            ),
            "type": "invalid_request_error",
            "param": "metadata.langgraph_thread_id",
            "code": None,
        }
    }


def test_streaming_interrupt_chunks_parse_with_openai_client(
    openai_client: OpenAI,
) -> None:
    chunks = list(create_completion(openai_client, stream=True))

    tool_call_chunks = [
        chunk
        for chunk in chunks
        if chunk.choices[0].delta.tool_calls is not None
    ]
    assert len(tool_call_chunks) == 1
    tool_call = tool_call_chunks[0].choices[0].delta.tool_calls[0]
    assert tool_call.index == 0
    assert tool_call.id is not None
    assert_interrupt_tool_call(tool_call)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_in_memory_saver_interrupt_resume_works_in_one_process(
    openai_client: OpenAI,
) -> None:
    first_response = create_completion(openai_client)
    final_response = resume_interrupt(
        openai_client,
        first_response,
        "approve",
    )

    assert interrupt_payload(first_response) == INTERRUPT_PAYLOAD
    assert final_response.choices[0].message.content == "resumed:approve"
