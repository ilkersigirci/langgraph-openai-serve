from importlib.metadata import version as metadata_version

from fastapi import FastAPI
from httpx import AsyncClient
from starlette import status


async def test_health_endpoint_returns_ok(
    client: AsyncClient,
    fastapi_app: FastAPI,
) -> None:
    url = fastapi_app.url_path_for("openai:health_check")

    response = await client.get(url)

    assert response.status_code == status.HTTP_200_OK


async def test_version_endpoint_reports_the_installed_distribution(
    client: AsyncClient,
    fastapi_app: FastAPI,
) -> None:
    url = fastapi_app.url_path_for("openai:version")

    response = await client.get(url)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "version": metadata_version("langgraph_openai_serve"),
    }
