import json

import pytest
from httpx import AsyncClient
from openai import AsyncOpenAI, BadRequestError
from starlette import status

pytestmark = pytest.mark.anyio


async def test_non_streaming_completion_matches_openai_contract(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.chat.completions.create(
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


async def test_streaming_completion_forwards_llm_chunks(
    openai_client: AsyncOpenAI,
) -> None:
    stream = await openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = [chunk async for chunk in stream]

    assert chunks[0].object == "chat.completion.chunk"
    assert chunks[0].model == "test"
    assert chunks[0].choices[0].delta.role == "assistant"
    content_deltas = [
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices[0].delta.content
    ]
    assert content_deltas == list("hello")
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_streaming_completion_uses_sse_wire_format(
    client: AsyncClient,
) -> None:
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

    assert events
    assert all(event.startswith("data: ") for event in events)
    assert events[-1] == "data: [DONE]"
    for event in events[:-1]:
        assert isinstance(json.loads(event.removeprefix("data: ")), dict)


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(False, id="non-streaming"),
        pytest.param(True, id="streaming"),
    ],
)
async def test_unknown_model_raises_openai_bad_request(
    openai_client: AsyncOpenAI,
    stream: bool,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model="missing",
            messages=[{"role": "user", "content": "Hi"}],
            stream=stream,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.response.json() == {
        "error": {
            "message": "Graph 'missing' not found in registry.",
            "type": "invalid_request_error",
            "param": "model",
            "code": None,
        }
    }
