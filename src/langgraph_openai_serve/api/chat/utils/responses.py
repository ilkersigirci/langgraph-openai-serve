"""OpenAI chat response builders."""

import json
import time
import uuid

from openai.types.shared import ErrorObject

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionStreamToolCall,
    ChatCompletionStreamToolCallFunction,
    Role,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)
from langgraph_openai_serve.api.chat.utils.interrupts import (
    INTERRUPT_TOOL_NAME,
    interrupt_arguments,
    interrupt_tool_call_id,
)
from langgraph_openai_serve.core.errors import openai_error_payload
from langgraph_openai_serve.graph.runner import LangGraphInterrupt, LangGraphOutput


def chat_completion_response(
    *,
    model: str,
    completion: LangGraphOutput,
    usage: dict[str, int],
) -> ChatCompletionResponse:
    """Build a non-streaming OpenAI-compatible chat completion response."""
    message, finish_reason = response_message(completion)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        ),
    )


def response_message(
    completion: LangGraphOutput,
) -> tuple[ChatCompletionResponseMessage, str]:
    if isinstance(completion, LangGraphInterrupt):
        return (
            ChatCompletionResponseMessage(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[interrupt_tool_call(completion)],
            ),
            "tool_calls",
        )

    return (
        ChatCompletionResponseMessage(
            role=Role.ASSISTANT,
            content=completion,
        ),
        "stop",
    )


def interrupt_tool_call(interrupt: LangGraphInterrupt) -> ToolCall:
    return ToolCall(
        id=interrupt_tool_call_id(interrupt.interrupt_id),
        type="function",
        function=ToolCallFunction(
            name=INTERRUPT_TOOL_NAME,
            arguments=interrupt_tool_arguments(interrupt),
        ),
    )


def interrupt_tool_arguments(interrupt: LangGraphInterrupt) -> str:
    return interrupt_arguments(
        thread_id=interrupt.thread_id,
        interrupt_id=interrupt.interrupt_id,
        payload=interrupt.payload,
    )


class ChatCompletionStreamResponseBuilder:
    """Build OpenAI-compatible chat completion SSE chunks."""

    def __init__(self, model: str) -> None:
        self.response_id = f"chatcmpl-{uuid.uuid4()}"
        self.created = int(time.time())
        self.model = model

    def role(self) -> str:
        return self._chunk(ChatCompletionStreamResponseDelta(role=Role.ASSISTANT))

    def text(self, content: str) -> str:
        return self._chunk(ChatCompletionStreamResponseDelta(content=content))

    def interrupt(self, interrupt: LangGraphInterrupt) -> str:
        return self._chunk(
            ChatCompletionStreamResponseDelta(
                tool_calls=[
                    ChatCompletionStreamToolCall(
                        index=0,
                        id=interrupt_tool_call_id(interrupt.interrupt_id),
                        type="function",
                        function=ChatCompletionStreamToolCallFunction(
                            name=INTERRUPT_TOOL_NAME,
                            arguments=interrupt_tool_arguments(interrupt),
                        ),
                    )
                ],
            ),
        )

    def finish(self, finish_reason: str) -> str:
        return self._chunk(
            ChatCompletionStreamResponseDelta(),
            finish_reason=finish_reason,
        )

    def error(self, message: str) -> str:
        return self._format_data(
            openai_error_payload(ErrorObject(message=message, type="server_error"))
        )

    def done(self) -> str:
        return "data: [DONE]\n\n"

    def _chunk(
        self,
        delta: ChatCompletionStreamResponseDelta,
        finish_reason: str | None = None,
    ) -> str:
        response = ChatCompletionStreamResponse(
            id=self.response_id,
            created=self.created,
            model=self.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )
        return self._format_data(response.model_dump())

    def _format_data(self, data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"
