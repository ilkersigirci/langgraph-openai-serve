from langchain_core.messages import HumanMessage

from langgraph_openai_serve import GraphRequest, NamedFunctionToolChoice
from langgraph_openai_serve.api.chat.request import decode_chat_request
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest


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

    graph_request, messages = decode_chat_request(request)

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
