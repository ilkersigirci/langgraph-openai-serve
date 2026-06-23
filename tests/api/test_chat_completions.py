import json

import pytest
from httpx import AsyncClient
from openai import OpenAI
from starlette import status


def test_create_chat_completion(openai_client: OpenAI) -> None:
    response = openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response.object == "chat.completion"
    assert response.model == "test"
    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content == "hello"
    assert choice.finish_reason == "stop"

    usage = response.usage
    assert usage is not None
    assert usage.prompt_tokens == 1
    assert usage.completion_tokens == 1
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_stream_chat_completion(openai_client: OpenAI) -> None:
    stream = openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = list(stream)

    assert chunks[0].object == "chat.completion.chunk"
    assert chunks[0].model == "test"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.anyio
async def test_stream_chat_completion_sse_wire_format(client: AsyncClient) -> None:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"].startswith("text/event-stream")

        events = []
        async for line in response.aiter_lines():
            if line:
                events.append(line)

    assert all(event.startswith("data: ") for event in events)
    payloads = [
        json.loads(event.removeprefix("data: "))
        for event in events
        if event != "data: [DONE]"
    ]
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(
        payload["choices"][0]["delta"].get("content") or "" for payload in payloads
    ) == "hello"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]"
