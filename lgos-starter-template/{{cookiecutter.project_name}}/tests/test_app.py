import pytest
from openai import AsyncOpenAI

from tests.support import ANSWER


async def test_registered_models_advertise_their_features(
    openai_client: AsyncOpenAI,
) -> None:
    models = await openai_client.models.list()
    assert {model.id for model in models.data} == {"simple-graph", "approval"}
    simple = await openai_client.models.retrieve("simple-graph")
    extension = simple.model_extra["lgos"]
    assert extension["features"] == ["background"]
    assert extension["client_settings"]["defaults"] == {
        "use_history": True,
        "audience": "general",
    }
    approval = await openai_client.models.retrieve("approval")
    assert set(approval.model_extra["lgos"]["features"]) == {"interrupts", "background"}


@pytest.mark.parametrize("stream", [False, True], ids=["response", "stream"])
async def test_responses_returns_graph_answer(
    openai_client: AsyncOpenAI, stream: bool
) -> None:
    response = await openai_client.responses.create(
        model="simple-graph", input="Hello", store=False, stream=stream
    )
    if stream:
        text = ""
        terminal = None
        async with response:
            async for event in response:
                if event.type == "response.output_text.delta":
                    text += event.delta
                elif event.type == "response.completed":
                    terminal = event.response
        assert text == ANSWER
        assert terminal is not None
        assert terminal.output_text == text
    else:
        assert response.output_text == ANSWER


@pytest.mark.parametrize("stream", [False, True], ids=["completion", "stream"])
async def test_chat_completions_returns_graph_answer(
    openai_client: AsyncOpenAI, stream: bool
) -> None:
    response = await openai_client.chat.completions.create(
        model="simple-graph",
        messages=[{"role": "user", "content": "Hello"}],
        stream=stream,
    )
    if stream:
        text = ""
        async with response:
            async for chunk in response:
                text += "".join(choice.delta.content or "" for choice in chunk.choices)
        assert text == ANSWER
    else:
        assert response.choices[0].message.content == ANSWER
