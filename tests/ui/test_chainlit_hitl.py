import json

from demo.ui.chainlit_ui.hitl import (
    assistant_tool_call_message,
    interrupt_payload,
    interrupt_tool_call,
)
from openai.types.chat import ChatCompletionMessage

from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME


def test_chainlit_hitl_helpers_use_openai_tool_call_models() -> None:
    payload = {
        "question": "Approve?",
        "request": "Refund ORDER-123.",
        "choices": ["approve", "reject"],
    }
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_other",
                "type": "function",
                "function": {"name": "other_tool", "arguments": "{}"},
            },
            {
                "id": "call_interrupt",
                "type": "function",
                "function": {
                    "name": INTERRUPT_TOOL_NAME,
                    "arguments": json.dumps({"payload": payload}),
                },
            },
        ],
    )

    tool_call = interrupt_tool_call(message)

    assert tool_call is not None
    assert tool_call.id == "call_interrupt"
    assert interrupt_payload(tool_call) == payload
    assert assistant_tool_call_message(message, tool_call) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_interrupt",
                "type": "function",
                "function": {
                    "name": INTERRUPT_TOOL_NAME,
                    "arguments": json.dumps({"payload": payload}),
                },
            }
        ],
    }
