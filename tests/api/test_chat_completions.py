from openai import OpenAI


def test_create_chat_completion(openai_client: OpenAI) -> None:
    response = openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response.object == "chat.completion"
    assert response.model == "test"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage
    assert response.usage.total_tokens == 2


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
