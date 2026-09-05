"""OpenAI Responses codec for LangGraph interrupt continuations."""

from langgraph_openai_serve.api.responses.schemas import (
    ResponseFunctionCallInput,
    ResponseFunctionCallOutputInput,
    ResponseInputItem,
)
from langgraph_openai_serve.graph.interrupt.codec import (
    INTERRUPT_TOOL_NAME,
    InterruptToolCall,
    InterruptToolOutput,
    is_interrupt_tool_call_id,
    parse_interrupt_exchange,
)
from langgraph_openai_serve.graph.interrupt.errors import InvalidResumeRequestError
from langgraph_openai_serve.graph.interrupt.models import InterruptResume


def parse_responses_resume(
    input_value: str | list[ResponseInputItem],
) -> InterruptResume | None:
    """Parse a trailing canonical interrupt function call/output exchange."""
    if isinstance(input_value, str):
        return None

    output_start = _trailing_output_start(input_value)
    if output_start is None:
        return None

    outputs = input_value[output_start:]
    call_start = output_start
    while call_start > 0 and isinstance(
        input_value[call_start - 1],
        ResponseFunctionCallInput,
    ):
        call_start -= 1
    calls = input_value[call_start:output_start]

    if not calls:
        if any(_is_interrupt_output(item) for item in outputs):
            msg = "Interrupt function outputs must follow their function calls."
            raise InvalidResumeRequestError(msg)
        return None

    interrupt_calls = [
        item
        for item in calls
        if isinstance(item, ResponseFunctionCallInput)
        and item.name == INTERRUPT_TOOL_NAME
    ]
    if not interrupt_calls:
        if any(_is_interrupt_output(item) for item in outputs):
            msg = "Interrupt function outputs must follow their function calls."
            raise InvalidResumeRequestError(msg)
        return None
    if len(interrupt_calls) != len(calls):
        msg = "Interrupt and ordinary function calls cannot be resumed together."
        raise InvalidResumeRequestError(msg)

    return parse_interrupt_exchange(
        [
            InterruptToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
            for call in interrupt_calls
        ],
        [
            InterruptToolOutput(call_id=output.call_id, output=output.output)
            for output in outputs
            if isinstance(output, ResponseFunctionCallOutputInput)
        ],
    )


def _trailing_output_start(items: list[ResponseInputItem]) -> int | None:
    if not items or not isinstance(items[-1], ResponseFunctionCallOutputInput):
        return None
    index = len(items) - 1
    while index > 0 and isinstance(
        items[index - 1],
        ResponseFunctionCallOutputInput,
    ):
        index -= 1
    return index


def _is_interrupt_output(item: ResponseInputItem) -> bool:
    return isinstance(item, ResponseFunctionCallOutputInput) and (
        is_interrupt_tool_call_id(item.call_id)
    )


__all__ = ["parse_responses_resume"]
