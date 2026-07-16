from typing import Any

import pytest
from demo.api.graphs import simple as simple_module
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("use_history", "expected_messages"),
    [
        pytest.param(
            True,
            [
                ("system", simple_module.DEFAULT_SYSTEM_PROMPT),
                ("human", "First"),
                ("ai", "Prior answer"),
                ("human", "Latest"),
            ],
            id="history",
        ),
        pytest.param(
            False,
            [
                ("system", simple_module.DEFAULT_SYSTEM_PROMPT),
                ("human", "Latest"),
            ],
            id="latest-message",
        ),
    ],
)
async def test_runtime_context_controls_conversation_history(
    monkeypatch: pytest.MonkeyPatch,
    use_history: bool,
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
        context=simple_module.SimpleContext(use_history=use_history),
    )

    assert [(message.type, message.content) for message in model_inputs[0]] == (
        expected_messages
    )
    assert result["messages"][-1].content == "Fake answer"
