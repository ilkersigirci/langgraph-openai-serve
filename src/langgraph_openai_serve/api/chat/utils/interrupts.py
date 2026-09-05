"""OpenAI Chat Completions codec for LangGraph interrupts."""

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequestMessage,
    Role,
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


def parse_resume_request(
    messages: list[ChatCompletionRequestMessage],
) -> InterruptResume | None:
    """
    Parse the trailing canonical assistant/tool interrupt exchange.

    Ordinary tool messages remain ordinary graph input. A LangGraph resume is
    recognized only when the tool results answer a preceding assistant message
    whose function calls are all ``langgraph_interrupt`` calls.
    """
    tool_start = _trailing_tool_start(messages)
    if tool_start is None:
        return None

    tool_messages = messages[tool_start:]
    assistant = messages[tool_start - 1] if tool_start > 0 else None

    if assistant is None or assistant.role != Role.ASSISTANT:
        if any(_is_interrupt_tool_result(message) for message in tool_messages):
            msg = "Interrupt tool results must follow their assistant tool calls."
            raise InvalidResumeRequestError(msg)
        return None

    calls = assistant.tool_calls or []
    interrupt_calls = [
        call for call in calls if call.function.name == INTERRUPT_TOOL_NAME
    ]

    if not interrupt_calls:
        if any(_is_interrupt_tool_result(message) for message in tool_messages):
            msg = "Interrupt tool results must follow their assistant tool calls."
            raise InvalidResumeRequestError(msg)
        return None

    if len(interrupt_calls) != len(calls):
        msg = "Interrupt and ordinary tool calls cannot be resumed in one exchange."
        raise InvalidResumeRequestError(msg)

    return parse_interrupt_exchange(
        [
            InterruptToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in interrupt_calls
        ],
        [
            InterruptToolOutput(
                call_id=_required_tool_call_id(message),
                output=message.content,
            )
            for message in tool_messages
        ],
    )


def _trailing_tool_start(
    messages: list[ChatCompletionRequestMessage],
) -> int | None:
    if not messages or messages[-1].role != Role.TOOL:
        return None

    index = len(messages) - 1
    while index > 0 and messages[index - 1].role == Role.TOOL:
        index -= 1
    return index


def _is_interrupt_tool_result(message: ChatCompletionRequestMessage) -> bool:
    return bool(
        message.tool_call_id and is_interrupt_tool_call_id(message.tool_call_id)
    )


def _required_tool_call_id(message: ChatCompletionRequestMessage) -> str:
    if not message.tool_call_id:
        msg = "Interrupt resume tool messages must include tool_call_id."
        raise InvalidResumeRequestError(msg)
    return message.tool_call_id


__all__ = ["parse_resume_request"]
