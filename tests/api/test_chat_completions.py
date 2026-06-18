from openai import OpenAI


def test_official_openai_client(openai_client: OpenAI) -> None:
    response = openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
    )
    stream = openai_client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    assert response.choices[0].message.content == "hello"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "hello"
