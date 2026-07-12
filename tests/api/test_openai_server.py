import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette import status

from langgraph_openai_serve import (
    GraphRegistry,
    GraphRegistryError,
    LanggraphOpenaiServe,
    openai_server,
)
from langgraph_openai_serve.core.settings import Settings


def _bind_test_app(
    graph_registry: GraphRegistry,
    *,
    prefix: str | None = None,
) -> FastAPI:
    return LanggraphOpenaiServe(
        graphs=graph_registry,
    ).bind_openai_api(prefix=prefix).app


def test_server_without_graphs_raises_registry_error() -> None:
    with pytest.raises(GraphRegistryError, match="at least one graph"):
        LanggraphOpenaiServe()


def test_empty_graph_registry_raises_registry_error() -> None:
    with pytest.raises(GraphRegistryError, match="at least one graph"):
        GraphRegistry(registry={})


def test_bind_openai_api_uses_settings_prefix_by_default(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_PREFIX", "/openai/v1")

    app = _bind_test_app(graph_registry)

    with TestClient(app) as client:
        response = client.get("/openai/v1/models")

    assert response.status_code == status.HTTP_200_OK


def test_bind_openai_api_explicit_prefix_overrides_settings(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_PREFIX", "/openai/v1")

    app = _bind_test_app(graph_registry, prefix="/v1")

    with TestClient(app) as client:
        configured_response = client.get("/v1/models")
        settings_response = client.get("/openai/v1/models")

    assert configured_response.status_code == status.HTTP_200_OK
    assert settings_response.status_code == status.HTTP_404_NOT_FOUND


def test_bind_openai_api_normalizes_explicit_prefix(
    graph_registry: GraphRegistry,
) -> None:
    app = _bind_test_app(graph_registry, prefix="/openai/v1/")

    with TestClient(app) as client:
        response = client.get("/openai/v1/models")

    assert response.status_code == status.HTTP_200_OK


def test_bind_openai_api_rejects_invalid_explicit_prefix(
    graph_registry: GraphRegistry,
) -> None:
    server = LanggraphOpenaiServe(graphs=graph_registry)

    with pytest.raises(ValueError):
        server.bind_openai_api(prefix="openai/v1")


def test_openai_api_docs_follow_settings(
    graph_registry: GraphRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_server.settings, "OPENAI_API_DOCS_ENABLED", False)
    disabled_app = _bind_test_app(graph_registry, prefix="/v1")

    with TestClient(disabled_app) as client:
        assert client.get("/v1/docs").status_code == status.HTTP_404_NOT_FOUND
        assert client.get("/v1/redoc").status_code == status.HTTP_404_NOT_FOUND
        assert client.get("/v1/openapi.json").status_code == status.HTTP_404_NOT_FOUND

    monkeypatch.setattr(openai_server.settings, "OPENAI_API_DOCS_ENABLED", True)
    enabled_app = _bind_test_app(graph_registry, prefix="/v1")

    with TestClient(enabled_app) as client:
        docs_response = client.get("/v1/docs")
        redoc_response = client.get("/v1/redoc")
        openapi_response = client.get("/v1/openapi.json")

    assert docs_response.status_code == status.HTTP_200_OK
    assert redoc_response.status_code == status.HTTP_200_OK
    assert openapi_response.status_code == status.HTTP_200_OK
    assert "/models" in openapi_response.json()["paths"]
    assert openapi_response.json()["servers"] == [{"url": "/v1"}]


def test_openai_api_prefix_settings_validation() -> None:
    settings = Settings(OPENAI_API_PREFIX="/openai/v1/")

    assert settings.OPENAI_API_PREFIX == "/openai/v1"

    for prefix in ["openai/v1", "///"]:
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_PREFIX=prefix)
