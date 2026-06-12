from fastapi.testclient import TestClient
from openai import OpenAI

from langgraph_openai_serve import (
    GraphConfig,
    GraphRegistry,
    LangchainOpenaiApiServe,
)


def test_official_openai_client(message_graph) -> None:
    registry = GraphRegistry(
        registry={
            "test": GraphConfig(
                graph=message_graph,
                streamable_node_names=["generate"],
            )
        }
    )
    app = LangchainOpenaiApiServe(graphs=registry).bind_openai_chat_completion().app

    with TestClient(app) as http_client:
        client = OpenAI(
            api_key="test",
            base_url="http://testserver/v1",
            http_client=http_client,
        )

        response = client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        stream = client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert response.choices[0].message.content == "hello"
        assert (
            "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "hello"
        )
