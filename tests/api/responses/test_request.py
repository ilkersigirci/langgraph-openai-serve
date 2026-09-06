import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_openai_serve import GraphRequest, NamedFunctionToolChoice
from langgraph_openai_serve.api.responses.interrupts import (
    interrupt_response_id,
    interrupt_tool_call_id,
)
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_responses_request,
    validate_hosted_tools,
)
from langgraph_openai_serve.api.responses.schemas import (
    ResponseCreateRequest,
    ResponseFunctionCallOutputInput,
    ResponseFunctionTool,
    ResponseHostedTool,
    ResponseNamedToolChoice,
)

EXPECTED_MESSAGE_COUNT = 2


def test_decode_responses_request_normalized_graph_inputs() -> None:
    request = ResponseCreateRequest(
        model="test-model",
        input="What is the weather?",
        instructions="Be concise.",
        metadata={"session_id": "session-1"},
        user="user-1",
        tools=[
            ResponseFunctionTool(
                name="get_weather",
                description="Get the weather.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                strict=True,
            ),
            ResponseHostedTool(name="lgos_time"),
        ],
        tool_choice=ResponseNamedToolChoice(type="function", name="get_weather"),
        parallel_tool_calls=False,
    )

    graph_request, messages, resume = decode_responses_request(request)

    assert isinstance(graph_request, GraphRequest)
    assert graph_request.model == "test-model"
    assert graph_request.metadata == {"session_id": "session-1"}
    assert graph_request.user == "user-1"
    assert len(graph_request.tools) == 1
    assert graph_request.tools[0].name == "get_weather"
    assert graph_request.tools[0].description == "Get the weather."
    assert graph_request.tools[0].strict is True
    assert graph_request.hosted_tools == ("lgos_time",)
    assert graph_request.tool_choice == NamedFunctionToolChoice(name="get_weather")
    assert graph_request.parallel_tool_calls is False
    assert len(messages) == EXPECTED_MESSAGE_COUNT
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Be concise."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "What is the weather?"
    assert resume is None


def test_decode_responses_request_with_previous_response_id() -> None:
    run_id = "11111111-1111-4111-8111-111111111111"
    state_token = "a" * 64
    request = ResponseCreateRequest(
        model="interruptible",
        previous_response_id=interrupt_response_id(run_id),
        input=[
            ResponseFunctionCallOutputInput(
                type="function_call_output",
                call_id=interrupt_tool_call_id("1", state_token),
                output="yes",
            ),
        ],
    )

    _graph_request, messages, resume = decode_responses_request(request)

    assert resume is not None
    assert resume.run_id == run_id
    assert resume.state_token == state_token
    assert resume.values == {"1": "yes"}
    assert messages == []


@pytest.mark.parametrize(
    ("field", "value", "param"),
    [
        ("store", True, "store"),
        ("background", True, "background"),
        ("conversation", "conv_123", "conversation"),
    ],
)
def test_decode_responses_request_rejects_unsupported_stateful_fields(
    field: str, value: object, param: str
) -> None:
    kwargs = {"model": "test", "input": "hi", field: value}
    request = ResponseCreateRequest(**kwargs)
    with pytest.raises(UnsupportedResponsesRequestError) as exc_info:
        decode_responses_request(request)
    assert exc_info.value.param == param


def test_validate_hosted_tools_rejects_unsupported() -> None:
    request = ResponseCreateRequest(
        model="test-model",
        input="hi",
        tools=[ResponseHostedTool(name="lgos_unsupported")],
    )
    with pytest.raises(UnsupportedResponsesRequestError) as exc_info:
        validate_hosted_tools(request, supported={"lgos_supported"})
    assert exc_info.value.param == "tools.0.name"
