import json
from functools import partial
from typing import Any

import httpx2
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from lgos_demo_api.core.settings import settings
from lgos_demo_api.graphs import simple as simple_module


@pytest.mark.parametrize(
    ("context", "expected_messages"),
    [
        pytest.param(
            simple_module.SimpleContext(use_history=True, audience="beginner"),
            [
                (
                    "system",
                    (
                        f"{simple_module.DEFAULT_SYSTEM_PROMPT} "
                        "Adapt explanations for beginner readers."
                    ),
                ),
                ("human", "First"),
                ("ai", "Prior answer"),
                ("human", "Latest"),
            ],
            id="history",
        ),
        pytest.param(
            simple_module.SimpleContext(use_history=False, audience="expert"),
            [
                (
                    "system",
                    (
                        f"{simple_module.DEFAULT_SYSTEM_PROMPT} "
                        "Adapt explanations for expert readers."
                    ),
                ),
                ("human", "Latest"),
            ],
            id="latest-message",
        ),
    ],
)
async def test_runtime_context_controls_model_input(
    monkeypatch: pytest.MonkeyPatch,
    context: simple_module.SimpleContext,
    expected_messages: list[tuple[str, str]],
) -> None:
    model_inputs: list[Any] = []

    async def respond(messages: Any) -> AIMessage:
        model_inputs.append(messages)
        return AIMessage(
            content="Fake answer",
            response_metadata={"finish_reason": "length"},
            usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    monkeypatch.setattr(
        "lgos_demo_api.utils.models.ChatOpenAI",
        lambda **_: RunnableLambda(respond),
    )

    result = await simple_module.simple_graph.ainvoke(
        simple_module.AgentState(
            messages=[
                HumanMessage(content="First"),
                AIMessage(content="Prior answer"),
                HumanMessage(content="Latest"),
            ],
        ),
        context=context,
    )

    assert [(message.type, message.content) for message in model_inputs[0]] == (
        expected_messages
    )
    assert result["messages"][-1].content == "Fake answer"
    assert result["messages"][-1].response_metadata == {"finish_reason": "length"}
    assert result["messages"][-1].usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }


async def test_selected_model_answers_through_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = settings.OPENAI_RESPONSES_MODEL
    monkeypatch.setattr(settings, "OPENAI_GATEWAY_BASE_URL", "https://gateway.example")
    monkeypatch.setattr(settings, "OPENAI_GATEWAY_API_KEY", "test-key")
    requests = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as provider:
        monkeypatch.setattr(
            "lgos_demo_api.utils.models.ChatOpenAI",
            partial(ChatOpenAI, http_async_client=provider),
        )
        result = await simple_module.simple_graph.ainvoke(
            simple_module.AgentState(messages=[HumanMessage(content="Hello")]),
            context=simple_module.SimpleContext(model=model_name),
        )

    [request] = requests
    assert str(request.url) == "https://gateway.example/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == model_name
    # Reasoning models reject non-default sampling, so leave it to the provider.
    assert "temperature" not in payload
    assert result["messages"][-1].content == "Hello."
