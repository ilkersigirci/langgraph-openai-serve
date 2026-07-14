from typing import Any

import pytest
from demo.api.graphs import simple as simple_module
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("use_history", "expected_prompt"),
    [
        pytest.param(
            True,
            [("human", "First"), ("ai", "Prior answer"), ("human", "Latest")],
            id="history",
        ),
        pytest.param(False, [("human", "Latest")], id="latest-message"),
    ],
)
async def test_config_controls_conversation_history(
    monkeypatch: pytest.MonkeyPatch,
    use_history: bool,
    expected_prompt: list[tuple[str, str]],
) -> None:
    prompts: list[Any] = []

    async def respond(prompt: Any) -> AIMessage:
        prompts.append(prompt)
        return AIMessage(content="Fake answer")

    monkeypatch.setattr(
        simple_module,
        "ChatOpenAI",
        lambda **_: RunnableLambda(respond),
    )

    result = await simple_module.simple_graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="First"),
                AIMessage(content="Prior answer"),
                HumanMessage(content="Latest"),
            ]
        },
        config={"configurable": {"use_history": use_history}},
    )

    prompt_messages = prompts[0].to_messages()[1:]
    assert [(message.type, message.content) for message in prompt_messages] == (
        expected_prompt
    )
    assert result["messages"][-1].content == "Fake answer"
