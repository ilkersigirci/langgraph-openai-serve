import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI, BadRequestError, NotFoundError
from starlette import status

from langgraph_openai_serve import GraphRegistry, LanggraphOpenaiServe

pytestmark = pytest.mark.anyio


async def test_validation_error_returns_openai_error(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.post(
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


async def test_empty_messages_returns_openai_error(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.post(
            "/chat/completions",
            cast_to=object,
            body={"model": "test", "messages": []},
        )

    response = exc_info.value.response
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages"
    assert error["code"] is None
    assert error["message"].startswith("messages: ")


async def test_missing_tool_call_id_returns_openai_error(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.post(
            "/chat/completions",
            cast_to=object,
            body={
                "model": "test",
                "messages": [{"role": "tool", "content": "Tool result"}],
            },
        )

    response = exc_info.value.response
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {
        "error": {
            "message": "Tool messages require the 'tool_call_id' field.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": None,
        }
    }


@pytest.mark.parametrize(
    ("body", "expected_param", "expected_message"),
    [
        pytest.param(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "functions": [],
            },
            "functions",
            "'functions' is not supported; use 'tools' instead.",
            id="functions",
        ),
        pytest.param(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "function_call": "auto",
            },
            "function_call",
            "'function_call' is not supported; use 'tool_choice' instead.",
            id="request-function-call",
        ),
        pytest.param(
            {
                "model": "test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            "messages.0.function_call",
            "'function_call' is not supported; use 'tool_calls' instead.",
            id="message-function-call",
        ),
        pytest.param(
            {
                "model": "test",
                "messages": [
                    {"role": "function", "name": "lookup", "content": "result"}
                ],
            },
            "messages.0.role",
            "Input should be 'system', 'user', 'assistant' or 'tool'",
            id="function-role",
        ),
    ],
)
async def test_legacy_function_calling_fields_are_rejected(
    openai_client: AsyncOpenAI,
    body: dict[str, object],
    expected_param: str | None,
    expected_message: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.post(
            "/chat/completions",
            cast_to=object,
            body=body,
        )

    response = exc_info.value.response
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == expected_param
    assert expected_message in error["message"]


async def test_metadata_pair_limit_returns_openai_error(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={f"key-{index}": "value" for index in range(17)},
        )

    response = exc_info.value.response
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "metadata"
    assert error["message"].startswith("metadata: ")


async def test_http_error_returns_openai_error(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await openai_client.models.retrieve("missing")

    response = exc_info.value.response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {
        "error": {
            "message": "Graph 'missing' not found in registry.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


async def test_openai_error_handlers_do_not_replace_host_app_handlers(
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

    LanggraphOpenaiServe(
        app=app,
        graphs=graph_registry,
    ).bind_openai_api(prefix="/v1")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        outside_response = await client.get("/outside")
        openai_response = await client.get("/v1/models/missing")

    assert outside_response.status_code == status.HTTP_418_IM_A_TEAPOT
    assert outside_response.json() == {"detail": "host handler"}
    assert openai_response.status_code == status.HTTP_404_NOT_FOUND
    assert openai_response.json() == {
        "error": {
            "message": "Graph 'missing' not found in registry.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }
