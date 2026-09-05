import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from openai import AsyncOpenAI, BadRequestError, InternalServerError
from starlette import status

from langgraph_openai_serve import GraphConfig, GraphRegistry, GraphRequest
from langgraph_openai_serve.graph.graph_registry import GraphConfigurationError
from tests.graph.support.message import make_message_graph


@pytest.mark.parametrize(
    "store_options",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"store": False}, id="false"),
    ],
)
async def test_async_openai_creates_stateless_text_response(
    openai_client: AsyncOpenAI,
    store_options: dict[str, bool],
) -> None:
    response = await openai_client.responses.create(
        model="test",
        input="Hi",
        text={"format": {"type": "text"}},
        **store_options,
    )

    assert response.object == "response"
    assert response.status == "completed"
    assert response.model == "test"
    assert response.output_text == "hello"
    assert response.background is False
    assert response.completed_at == response.created_at
    assert response.usage is None
    assert response.text is not None
    assert response.text.format is not None
    assert response.text.format.type == "text"
    assert (response.model_extra or {})["store"] is False

    message = response.output[0]
    assert message.type == "message"
    assert message.status == "completed"
    assert message.phase == "final_answer"
    assert message.content[0].type == "output_text"
    assert message.content[0].annotations == []


async def test_message_input_preserves_order_roles_and_replay_metadata(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received_requests: list[GraphRequest] = []
    received_messages: list[list[BaseMessage]] = []

    def capture_input(
        request: GraphRequest,
        messages: list[BaseMessage],
    ) -> dict[str, list[BaseMessage]]:
        received_requests.append(request)
        received_messages.append(messages)
        return {"messages": messages}

    graph_registry.get_graph("test").request_to_input = capture_input

    response = await openai_client.responses.create(
        model="test",
        instructions="Follow the instruction.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "First."},
                    {"type": "input_text", "text": "Second."},
                ],
            },
            {"role": "developer", "content": "Developer context."},
            {
                "role": "assistant",
                "content": "Working.",
                "phase": "commentary",
            },
            {
                "id": "msg_prior",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Prior answer.",
                        "annotations": [],
                    }
                ],
            },
        ],
        metadata={"session_id": "session-1"},
        user="alice",
        store=False,
    )

    assert response.output_text == "hello"
    assert response.instructions == "Follow the instruction."
    assert response.metadata == {"session_id": "session-1"}
    assert response.user == "alice"
    assert received_requests == [
        GraphRequest(
            model="test",
            metadata={"session_id": "session-1"},
            user="alice",
            tools=(),
            tool_choice=None,
            parallel_tool_calls=None,
        )
    ]

    messages = received_messages[0]
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        SystemMessage,
        AIMessage,
        AIMessage,
    ]
    assert messages[0].text == "Follow the instruction."
    assert messages[1].content == [
        {"type": "text", "text": "First."},
        {"type": "text", "text": "Second."},
    ]
    assert messages[2].additional_kwargs == {"__openai_role__": "developer"}
    assert messages[3].additional_kwargs == {"phase": "commentary"}
    assert messages[4].id == "msg_prior"
    assert messages[4].additional_kwargs == {
        "id": "msg_prior",
        "phase": "final_answer",
    }


async def test_file_id_input_uses_the_protocol_neutral_graph_shape(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    received_messages: list[list[BaseMessage]] = []

    def capture_input(
        _request: GraphRequest,
        messages: list[BaseMessage],
    ) -> dict[str, list[BaseMessage]]:
        received_messages.append(messages)
        return {"messages": messages}

    graph_registry.get_graph("test").request_to_input = capture_input

    await openai_client.responses.create(
        model="test",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Summarize this."},
                    {"type": "input_file", "file_id": "file_report"},
                ],
            }
        ],
    )

    assert received_messages[0][0].content == [
        {"type": "text", "text": "Summarize this."},
        {"type": "file", "file": {"file_id": "file_report"}},
    ]


@pytest.mark.parametrize(
    "file_part",
    [
        {"file_url": "https://example.com/a.pdf"},
        {"file_data": "data:application/pdf;base64,AA=="},
    ],
)
async def test_unimplemented_file_sources_are_rejected(
    openai_client: AsyncOpenAI,
    file_part: dict[str, str],
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_file", **file_part}],
                }
            ],
        )

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"].startswith("input.")


async def test_empty_final_text_remains_a_completed_message(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    graph_registry.register(
        "empty",
        GraphConfig(graph=make_message_graph(""), description="DUMMY"),
    )

    response = await openai_client.responses.create(model="empty", input="Hi")

    assert not response.output_text
    assert response.output[0].type == "message"
    assert not response.output[0].content[0].text


async def test_provider_usage_maps_to_responses_details(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    usage = {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    graph_registry.register(
        "usage",
        GraphConfig(
            graph=make_message_graph(),
            description="DUMMY",
            output_to_message=lambda _output: AIMessage(
                content="counted",
                usage_metadata=usage,
            ),
        ),
    )

    response = await openai_client.responses.create(model="usage", input="Hi")

    assert response.usage is not None
    assert response.usage.input_tokens == usage["input_tokens"]
    assert response.usage.output_tokens == usage["output_tokens"]
    assert response.usage.total_tokens == usage["total_tokens"]
    assert response.usage.input_tokens_details.cached_tokens == 0
    assert response.usage.input_tokens_details.cache_write_tokens == 0
    assert response.usage.output_tokens_details.reasoning_tokens == 0


@pytest.mark.parametrize(
    ("options", "expected_param", "message_part"),
    [
        pytest.param(
            {"store": True},
            "store",
            "'store' must be false",
            id="stored-response",
        ),
        pytest.param(
            {"background": True},
            "background",
            "Background Responses",
            id="background",
        ),
        pytest.param(
            {"conversation": "conv_test"},
            "conversation",
            "Responses conversations",
            id="conversation",
        ),
        pytest.param(
            {"previous_response_id": "resp_prior"},
            "previous_response_id",
            "Previous response state",
            id="previous-response",
        ),
    ],
)
async def test_stateful_options_are_rejected(
    openai_client: AsyncOpenAI,
    options: dict[str, object],
    expected_param: str,
    message_part: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Hi",
            **options,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == expected_param
    assert error["code"] is None
    assert message_part in error["message"]


@pytest.mark.parametrize(
    ("options", "expected_param"),
    [
        pytest.param({"temperature": 0.2}, "temperature", id="generation"),
        pytest.param(
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": {"type": "object"},
                    }
                }
            },
            "text.format.type",
            id="structured-output",
        ),
    ],
)
async def test_unimplemented_response_options_are_explicit_errors(
    openai_client: AsyncOpenAI,
    options: dict[str, object],
    expected_param: str,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(model="test", input="Hi", **options)

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == expected_param
    assert error["code"] is None


async def test_metadata_limits_apply_to_responses(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input="Hi",
            metadata={f"key-{index}": "value" for index in range(17)},
        )

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "metadata"


async def test_duplicate_replayed_message_ids_are_rejected(
    openai_client: AsyncOpenAI,
) -> None:
    replayed_message = {
        "id": "msg_duplicate",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [
            {
                "type": "output_text",
                "text": "Prior answer.",
                "annotations": [],
            }
        ],
    }

    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(
            model="test",
            input=[replayed_message, replayed_message],
        )

    error = exc_info.value.response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "input"
    assert "duplicate item id 'msg_duplicate'" in error["message"]


async def test_unknown_model_uses_openai_error_envelope(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        await openai_client.responses.create(model="missing", input="Hi")

    assert exc_info.value.response.json() == {
        "error": {
            "message": "Graph 'missing' not found in registry.",
            "type": "invalid_request_error",
            "param": "model",
            "code": None,
        }
    }


async def test_graph_configuration_error_uses_server_error_envelope(
    openai_client: AsyncOpenAI,
    graph_registry: GraphRegistry,
) -> None:
    def reject_output(_output: object) -> AIMessage:
        message = "Graph output is not configured."
        raise GraphConfigurationError(message)

    graph_registry.register(
        "broken",
        GraphConfig(
            graph=make_message_graph(),
            description="DUMMY",
            output_to_message=reject_output,
        ),
    )

    with pytest.raises(InternalServerError) as exc_info:
        await openai_client.responses.create(model="broken", input="Hi")

    assert exc_info.value.response.json() == {
        "error": {
            "message": "Graph output is not configured.",
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }
