from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from lgos_demo_api.graphs import simple as simple_module


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("context", "expected_messages"),
    [
        pytest.param(
            simple_module.SimpleContext(use_history=True, audience="beginner"),
            [
                (
                    "system",
                    f"{simple_module.DEFAULT_SYSTEM_PROMPT} "
                    "Adapt explanations for beginner readers.",
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
                    f"{simple_module.DEFAULT_SYSTEM_PROMPT} "
                    "Adapt explanations for expert readers.",
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
        return AIMessage(content="Fake answer")

    monkeypatch.setattr(
        simple_module,
        "ChatOpenAI",
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
