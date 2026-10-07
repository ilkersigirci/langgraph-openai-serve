import pytest
from openai import AsyncOpenAI, BadRequestError

from tests.support import ANSWER


async def test_registry_serves_each_graph_as_a_model(
    openai_client: AsyncOpenAI,
) -> None:
    models = await openai_client.models.list()
    assert {model.id for model in models.data} == {"simple-graph", "approval"}


async def test_simple_graph_rejects_invalid_runtime_settings(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as error:
        await openai_client.responses.create(
            model="simple-graph",
            input="Hello",
            metadata={"lgos_settings": '{"audience": "children"}'},
            store=False,
        )
    assert error.value.param == "metadata.lgos_settings"


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
