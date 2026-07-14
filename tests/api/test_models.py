import pytest
from openai import AsyncOpenAI

pytestmark = pytest.mark.anyio


async def test_registered_graphs_are_exposed_as_models(
    openai_client: AsyncOpenAI,
) -> None:
    response = await openai_client.models.list()

    assert response.object == "list"
    assert response.data[0].id == "test"
    assert response.data[0].object == "model"
