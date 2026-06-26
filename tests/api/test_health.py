import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from starlette import status


@pytest.mark.anyio
async def test_health(client: AsyncClient, fastapi_app: FastAPI) -> None:
    url = fastapi_app.url_path_for("openai:health_check")

    response = await client.get(url)

    assert response.status_code == status.HTTP_200_OK
