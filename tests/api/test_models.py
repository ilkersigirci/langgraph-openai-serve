from openai import OpenAI


def test_list_models(openai_client: OpenAI) -> None:
    response = openai_client.models.list()

    assert response.object == "list"
    assert response.data[0].id == "test"
    assert response.data[0].object == "model"
