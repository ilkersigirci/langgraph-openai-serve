import json

from langgraph.types import Interrupt

from langgraph_openai_serve.api.chat.utils.responses import (
    ChatCompletionStreamResponseBuilder,
    response_message,
)
from langgraph_openai_serve.graph.runner import LangGraphInterruptBatch

EXPECTED_TOOL_CALL_IDS = [
    "lg_interrupt_interrupt-b",
    "lg_interrupt_interrupt-a",
]
EXPECTED_ARGUMENTS = [
    {
        "run_id": "run-1",
        "state_token": "state-token-1",
        "payload": {"question": "B?"},
    },
    {
        "run_id": "run-1",
        "state_token": "state-token-1",
        "payload": {"question": "A?"},
    },
]


def _interrupt_batch() -> LangGraphInterruptBatch:
    return LangGraphInterruptBatch(
        run_id="run-1",
        state_token="state-token-1",
        interrupts=(
            Interrupt(id="interrupt-b", value={"question": "B?"}),
            Interrupt(id="interrupt-a", value={"question": "A?"}),
        ),
    )


def test_non_streaming_interrupt_batch_preserves_tool_call_order() -> None:
    message, finish_reason = response_message(_interrupt_batch())

    assert finish_reason == "tool_calls"
    assert message.content is None
    assert message.tool_calls is not None
    assert [tool_call.id for tool_call in message.tool_calls] == EXPECTED_TOOL_CALL_IDS
    assert [
        json.loads(tool_call.function.arguments) for tool_call in message.tool_calls
    ] == EXPECTED_ARGUMENTS


def test_streaming_interrupt_batch_uses_stable_tool_call_indices() -> None:
    event = ChatCompletionStreamResponseBuilder("interruptible").interrupt(
        _interrupt_batch()
    )
    payload = json.loads(event.removeprefix("data: ").strip())
    tool_calls = payload["choices"][0]["delta"]["tool_calls"]

    assert [tool_call["index"] for tool_call in tool_calls] == [0, 1]
    assert [tool_call["id"] for tool_call in tool_calls] == EXPECTED_TOOL_CALL_IDS
    assert [
        json.loads(tool_call["function"]["arguments"]) for tool_call in tool_calls
    ] == EXPECTED_ARGUMENTS
