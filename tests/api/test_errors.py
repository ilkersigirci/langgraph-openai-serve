import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from openai import BadRequestError, NotFoundError, OpenAI
from starlette import status

from langgraph_openai_serve import GraphRegistry, LangchainOpenaiApiServe


def test_validation_error_returns_openai_error(openai_client: OpenAI) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        openai_client.post(
            "/chat/completions",
            cast_to=object,
            body={"model": "test"},
        )

    response = exc_info.value.response
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {
        "error": {
            "message": "messages: Field required",
            "type": "invalid_request_error",
            "param": "messages",
            "code": None,
        }
    }


def test_http_error_returns_openai_error(openai_client: OpenAI) -> None:
    with pytest.raises(NotFoundError) as exc_info:
        openai_client.models.retrieve("missing")

    response = exc_info.value.response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {
        "error": {
            "message": "Not Found",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }


def test_openai_error_handlers_do_not_replace_host_app_handlers(
    graph_registry: GraphRegistry,
) -> None:
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def host_http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": "host handler"},
        )

    @app.get("/outside")
    def outside() -> None:
        raise HTTPException(status_code=status.HTTP_418_IM_A_TEAPOT)

    LangchainOpenaiApiServe(
        app=app,
        graphs=graph_registry,
    ).bind_openai_chat_completion(prefix="/v1")

    with TestClient(app) as client:
        outside_response = client.get("/outside")
        openai_response = client.get("/v1/models/missing")

    assert outside_response.status_code == status.HTTP_418_IM_A_TEAPOT
    assert outside_response.json() == {"detail": "host handler"}
    assert openai_response.status_code == status.HTTP_404_NOT_FOUND
    assert openai_response.json() == {
        "error": {
            "message": "Not Found",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }
