from collections.abc import Awaitable, Callable

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from httpx2 import ASGITransport, AsyncClient
from langchain_core.exceptions import ContextOverflowError
from langgraph.graph import StateGraph
from openai import AsyncOpenAI, BadRequestError, NotFoundError
from starlette import status

from langgraph_openai_serve import GraphConfig, GraphRegistry, LanggraphOpenaiServe
from tests.graph.support.schemas import MessageState

USER_MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    ("body", "param", "message"),
    [
        pytest.param({}, "messages", "messages: Field required", id="no-messages"),
        pytest.param(
            {"messages": []}, "messages", "at least 1 item", id="empty-messages"
        ),
        pytest.param(
            {"messages": [{"role": "tool", "content": "Tool result"}]},
            "messages",
            "Tool messages require the 'tool_call_id' field.",
            id="tool-message-without-call-id",
        ),
        pytest.param(
            {"messages": USER_MESSAGES, "functions": []},
            "functions",
            "Extra inputs are not permitted",
            id="deprecated-functions",
        ),
        pytest.param(
            {"messages": [{"role": "function", "name": "lookup", "content": "x"}]},
            "messages.0.role",
            "Input should be 'system', 'user', 'assistant' or 'tool'",
            id="function-role",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "Hello", "message_extra": 1}]},
            "messages.0.message_extra",
            "Extra inputs are not permitted",
            id="unknown-message-field",
        ),
        pytest.param(
            {"messages": USER_MESSAGES, "temperature": 0.2},
            "temperature",
            "Extra inputs are not permitted",
            id="generation-control",
        ),
        pytest.param(
            {"messages": USER_MESSAGES, "stream_options": {"include_usage": True}},
            None,
            "stream_options may only be set when stream is true",
            id="stream-options-without-stream",
        ),
        pytest.param(
            {
                "messages": USER_MESSAGES,
                "metadata": {f"key-{index}": "value" for index in range(17)},
            },
            "metadata",
            "at most 16 items",
            id="too-many-metadata-pairs",
        ),
    ],
)
async def test_unsupported_chat_requests_return_openai_errors(
    openai_client: AsyncOpenAI,
    body: dict[str, object],
    param: str | None,
    message: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.post(
            "/chat/completions", cast_to=object, body={"model": "test", **body}
        )

    error = exc_info.value.response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == (
        "invalid_request_error",
        param,
        None,
    )
    assert message in error["message"]


# The param names the rejected part, or its unsupported field.
@pytest.mark.parametrize(
    ("content_part", "param_end"),
    [
        pytest.param(
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ".0",
            id="image",
        ),
        pytest.param(
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
            ".0",
            id="audio",
        ),
        pytest.param(
            {
                "type": "file",
                "file": {
                    "file_id": "file-test",
                    "file_data": "data:application/pdf;base64,AA==",
                },
            },
            ".file_data",
            id="inline-file-data",
        ),
    ],
)
async def test_unsupported_chat_content_parts_are_rejected(
    openai_client: AsyncOpenAI,
    content_part: dict[str, object],
    param_end: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.post(
            "/chat/completions",
            cast_to=object,
            body={
                "model": "test",
                "messages": [{"role": "user", "content": [content_part]}],
            },
        )

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"].startswith("messages.0.content")
    assert error["param"].endswith(param_end)


@pytest.mark.parametrize(
    "create",
    [
        pytest.param(
            lambda client, end_user: client.chat.completions.create(
                model="test",
                messages=[{"role": "user", "content": "Hello"}],
                safety_identifier=end_user,
            ),
            id="chat",
        ),
        pytest.param(
            lambda client, end_user: client.responses.create(
                model="test", input="Hello", safety_identifier=end_user
            ),
            id="responses",
        ),
    ],
)
async def test_safety_identifier_over_openai_limit_returns_openai_error(
    openai_client: AsyncOpenAI,
    create: Callable[[AsyncOpenAI, str], Awaitable[object]],
) -> None:
    # OpenAI documents a 128-character maximum on both endpoints.
    await create(openai_client, "u" * 128)

    with pytest.raises(BadRequestError) as exc_info:
        await create(openai_client, "u" * 129)

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "safety_identifier"


async def test_http_error_returns_openai_error(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await openai_client.models.retrieve("missing")

    response = exc_info.value.response
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {
        "error": {
            "message": "The model 'missing' does not exist.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
            "misalignment": None,
        }
    }


@pytest.mark.parametrize(
    "create",
    [
        pytest.param(
            lambda client: client.chat.completions.create(
                model="overflow", messages=[{"role": "user", "content": "Hi"}]
            ),
            id="chat",
        ),
        pytest.param(
            lambda client: client.responses.create(model="overflow", input="Hi"),
            id="responses",
        ),
    ],
)
async def test_model_context_overflow_returns_non_retryable_openai_error(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
    create: Callable[[AsyncOpenAI], Awaitable[object]],
) -> None:
    def overflow(_state: MessageState) -> MessageState:
        msg = "This model's maximum context length is 8 tokens."
        raise ContextOverflowError(msg)

    graph_registry.graphs["overflow"] = GraphConfig(
        graph=StateGraph(MessageState)
        .add_node("overflow", overflow)
        .set_entry_point("overflow")
        .compile(),
        description="DUMMY",
    )

    with pytest.raises(BadRequestError) as exc_info:
        await create(openai_client)

    assert exc_info.value.response.json() == {
        "error": {
            "message": "The input exceeds the model's context window.",
            "type": "invalid_request_error",
            "param": None,
            "code": "context_length_exceeded",
            "misalignment": None,
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
        registry=graph_registry,
    ).bind_openai_api(prefix="/v1")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        outside_response = await client.get("/outside")
        openai_response = await client.get("/v1/models/missing")

    assert outside_response.status_code == status.HTTP_418_IM_A_TEAPOT
    assert outside_response.json() == {"detail": "host handler"}
    assert openai_response.status_code == status.HTTP_404_NOT_FOUND
    assert openai_response.json()["error"]["code"] == "model_not_found"
