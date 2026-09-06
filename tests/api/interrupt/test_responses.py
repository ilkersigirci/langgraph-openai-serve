"""Unit coverage for Responses API interrupt serialization."""

import json

from langgraph.types import Interrupt

from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.service import interrupt_output_items
from langgraph_openai_serve.api.responses.streaming import ResponsesStreamBuilder
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch

STATE_TOKEN = "a" * 64
EXPECTED_CALL_IDS = [
    f"call_lg_{STATE_TOKEN}_interrupt-b",
    f"call_lg_{STATE_TOKEN}_interrupt-a",
]
EXPECTED_ARGUMENTS = [
    {"question": "B?"},
    {"question": "A?"},
]


def _interrupt_batch() -> LangGraphInterruptBatch:
    return LangGraphInterruptBatch(
        run_id="run-1",
        state_token=STATE_TOKEN,
        interrupts=(
            Interrupt(id="interrupt-b", value={"question": "B?"}),
            Interrupt(id="interrupt-a", value={"question": "A?"}),
        ),
    )


def test_interrupt_output_items_preserves_order() -> None:
    calls = interrupt_output_items(_interrupt_batch())

    assert len(calls) == len(EXPECTED_CALL_IDS)
    assert [call.call_id for call in calls] == EXPECTED_CALL_IDS
    assert [call.name for call in calls] == [
        "langgraph_interrupt",
        "langgraph_interrupt",
    ]
    assert [json.loads(call.arguments) for call in calls] == EXPECTED_ARGUMENTS


def test_streaming_finish_interrupt_uses_stable_indices() -> None:
    request = ResponseCreateRequest(model="interruptible", input="Hi")
    builder = ResponsesStreamBuilder(request)
    events = list(builder.finish_interrupt(_interrupt_batch(), usage=None))

    added_events = [e for e in events if e.type == "response.output_item.added"]
    assert len(added_events) == len(EXPECTED_CALL_IDS)
    assert [e.output_index for e in added_events] == [0, 1]
    assert getattr(added_events[0].item, "call_id", None) == EXPECTED_CALL_IDS[0]
    assert getattr(added_events[1].item, "call_id", None) == EXPECTED_CALL_IDS[1]

    delta_events = [
        e for e in events if e.type == "response.function_call_arguments.delta"
    ]
    assert len(delta_events) == len(EXPECTED_CALL_IDS)
    assert [json.loads(e.delta) for e in delta_events] == EXPECTED_ARGUMENTS

    completed_events = [e for e in events if e.type == "response.completed"]
    assert len(completed_events) == 1
    assert completed_events[0].response.status == "completed"
