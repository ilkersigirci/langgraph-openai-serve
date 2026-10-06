import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from {{ cookiecutter.project_slug }}.graphs.simple import (
    SimpleContext,
    create_simple_graph,
)


@pytest.mark.parametrize("use_history", [False, True], ids=["latest", "history"])
async def test_runtime_settings_control_model_input(use_history: bool) -> None:
    prompts: list[BaseMessage] = []

    # A model callback observes the actual messages delivered by the graph.
    class CapturePrompt(AsyncCallbackHandler):
        async def on_chat_model_start(self, _serialized, messages, **_kwargs) -> None:
            prompts.extend(messages[0])

    graph = create_simple_graph(FakeListChatModel(responses=["Answer"]))
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="Earlier"),
                AIMessage(content="Prior"),
                HumanMessage(content="Latest"),
            ]
        },
        context=SimpleContext(use_history=use_history, audience="beginner"),
        config={"callbacks": [CapturePrompt()]},
    )
    assert (
        prompts[0].content
        == "You are a helpful assistant. Explain for beginner readers."
    )
    assert [message.content for message in prompts[1:]] == (
        ["Earlier", "Prior", "Latest"] if use_history else ["Latest"]
    )
