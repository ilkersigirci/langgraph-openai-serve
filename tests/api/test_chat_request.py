from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_openai_serve import GraphRequest, NamedFunctionToolChoice
from langgraph_openai_serve.api.chat.request import decode_chat_request
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.graph.interrupt.codec import (
    INTERRUPT_TOOL_NAME,
    interrupt_arguments,
    interrupt_tool_call_id,
)

RUN_ID = "11111111-1111-4111-8111-111111111111"


def test_chat_request_decodes_normalized_graph_inputs() -> None:
    request = ChatCompletionRequest(
        model="weather",
        messages=[{"role": "user", "content": "Weather?"}],
        metadata={"session_id": "session-1"},
        user="user-1",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    "strict": True,
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
        parallel_tool_calls=False,
    )

    graph_request, messages, resume = decode_chat_request(request)

    assert isinstance(graph_request, GraphRequest)
    assert graph_request.model == "weather"
    assert graph_request.metadata == {"session_id": "session-1"}
    assert graph_request.user == "user-1"
    assert graph_request.tools[0].name == "get_weather"
    assert graph_request.tools[0].description == "Get the weather."
    assert graph_request.tools[0].parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }
    assert graph_request.tools[0].strict is True
    assert graph_request.tool_choice == NamedFunctionToolChoice(name="get_weather")
    assert graph_request.parallel_tool_calls is False
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].text == "Weather?"
    assert resume is None


def test_chat_request_decodes_interrupt_resume_with_messages() -> None:
    tool_call_id = interrupt_tool_call_id("interrupt-1")
    request = ChatCompletionRequest(
        model="review",
        messages=[
            {"role": "user", "content": "Review this."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": INTERRUPT_TOOL_NAME,
                            "arguments": interrupt_arguments(
                                run_id=RUN_ID,
                                state_token="state-1",
                                payload={"question": "Approve?"},
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": '{"resume":"approve"}',
            },
        ],
    )

    _, messages, resume = decode_chat_request(request)

    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    assert resume is not None
    assert resume.run_id == RUN_ID
    assert resume.state_token == "state-1"
    assert resume.values == {"interrupt-1": "approve"}
