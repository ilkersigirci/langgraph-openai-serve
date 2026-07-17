import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from pydantic import ValidationError
from starlette import status

from langgraph_openai_serve import (
    GraphRegistry,
    LanggraphOpenaiServe,
    openai_server,
)
from langgraph_openai_serve.core.settings import Settings


def _bind_test_app(
    graph_registry: GraphRegistry,
    *,
    prefix: str | None = None,
) -> FastAPI:
    return (
        LanggraphOpenaiServe(
            graphs=graph_registry,
        )
        .bind_openai_api(prefix=prefix)
        .app
    )


async def _get(app: FastAPI, path: str) -> Response:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.anyio
async def test_bind_openai_api_uses_settings_prefix_by_default(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_PREFIX", "/openai/v1")

    app = _bind_test_app(graph_registry)

    response = await _get(app, "/openai/v1/models")

    assert response.status_code == status.HTTP_200_OK


@pytest.mark.anyio
async def test_bind_openai_api_explicit_prefix_overrides_settings(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_PREFIX", "/openai/v1")

    app = _bind_test_app(graph_registry, prefix="/v1")

    configured_response = await _get(app, "/v1/models")
    settings_response = await _get(app, "/openai/v1/models")

    assert configured_response.status_code == status.HTTP_200_OK
    assert settings_response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.anyio
async def test_bind_openai_api_normalizes_explicit_prefix(
    graph_registry: GraphRegistry,
) -> None:
    app = _bind_test_app(graph_registry, prefix="/openai/v1/")

    response = await _get(app, "/openai/v1/models")

    assert response.status_code == status.HTTP_200_OK


def test_bind_openai_api_rejects_invalid_explicit_prefix(
    graph_registry: GraphRegistry,
) -> None:
    server = LanggraphOpenaiServe(graphs=graph_registry)

    with pytest.raises(ValueError):
        server.bind_openai_api(prefix="openai/v1")


@pytest.mark.parametrize(
    ("enabled", "expected_status"),
    [
        pytest.param(False, status.HTTP_404_NOT_FOUND, id="disabled"),
        pytest.param(True, status.HTTP_200_OK, id="enabled"),
    ],
)
@pytest.mark.anyio
async def test_openai_api_docs_visibility_follows_settings(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    expected_status: int,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_DOCS_ENABLED", enabled)
    app = _bind_test_app(graph_registry, prefix="/v1")

    responses = [
        await _get(app, "/v1/docs"),
        await _get(app, "/v1/redoc"),
        await _get(app, "/v1/openapi.json"),
    ]

    assert [response.status_code for response in responses] == [expected_status] * 3


@pytest.mark.anyio
async def test_openai_api_schema_describes_mounted_api(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_DOCS_ENABLED", True)
    app = _bind_test_app(graph_registry, prefix="/v1")

    openapi_response = await _get(app, "/v1/openapi.json")

    assert openapi_response.status_code == status.HTTP_200_OK
    schema = openapi_response.json()
    assert "/models" in schema["paths"]
    assert "/models/{model}" in schema["paths"]
    assert schema["servers"] == [{"url": "/v1"}]


def test_openai_api_prefix_settings_normalizes_trailing_slash() -> None:
    settings = Settings(OPENAI_API_PREFIX="/openai/v1/")

    assert settings.OPENAI_API_PREFIX == "/openai/v1"


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("openai/v1", id="missing-leading-slash"),
        pytest.param("///", id="empty-after-normalization"),
    ],
)
def test_openai_api_prefix_settings_rejects_invalid_value(prefix: str) -> None:
    with pytest.raises(ValidationError):
        Settings(OPENAI_API_PREFIX=prefix)
