"""OpenAI chat response builders."""

import json
import time
import uuid

from langchain_core.messages import AIMessage, UsageMetadata
from langgraph.types import Interrupt
from openai.types.chat.chat_completion_message import Annotation
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
from langgraph_openai_serve.core.errors import openai_error_payload
from langgraph_openai_serve.graph.citations import citations_from_message
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.interrupt.codec import (
    INTERRUPT_TOOL_NAME,
    interrupt_arguments,
    interrupt_tool_call_id,
)
from langgraph_openai_serve.graph.runner import LangGraphOutput


def chat_completion_response(
    *,
    model: str,
    completion: LangGraphOutput,
) -> ChatCompletionResponse:
    """Build a non-streaming OpenAI-compatible chat completion response."""
    message, finish_reason = response_message(completion)
    usage = completion.usage_metadata if isinstance(completion, AIMessage) else None
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
        usage=usage_info(usage),
    )


def response_message(
    completion: LangGraphOutput,
) -> tuple[ChatCompletionResponseMessage, str]:
    """Format response message."""
    if isinstance(completion, LangGraphInterruptBatch):
        return (
            ChatCompletionResponseMessage(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    interrupt_tool_call(completion, interrupt)
                    for interrupt in completion.interrupts
                ],
            ),
            "tool_calls",
        )

    tool_calls = tool_calls_from_message(completion)
    return (
        ChatCompletionResponseMessage(
            role=Role.ASSISTANT,
            content=completion.text or None,
            annotations=annotations_from_message(completion) or None,
            tool_calls=tool_calls or None,
        ),
        "tool_calls" if tool_calls else "stop",
    )


def tool_calls_from_message(message: AIMessage) -> list[ToolCall]:
    """Convert native LangChain tool calls to Chat Completions tool calls."""
    tool_calls = []
    for tool_call in message.tool_calls:
        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            msg = "Final AIMessage tool calls must have an id."
            raise ValueError(msg)
        tool_calls.append(
            ToolCall(
                id=tool_call_id,
                function=ToolCallFunction(
                    name=tool_call["name"],
                    arguments=json.dumps(tool_call["args"]),
                ),
            )
        )
    return tool_calls


def annotations_from_message(message: AIMessage) -> list[Annotation]:
    """Convert validated LangChain citations to Chat URL annotations."""
    return [
        Annotation.model_validate(
            {
                "type": "url_citation",
                "url_citation": {
                    key: citation[key]
                    for key in ("url", "title", "start_index", "end_index")
                },
            }
        )
        for citation in citations_from_message(message)
    ]


def usage_info(usage: UsageMetadata | None) -> UsageInfo | None:
    """Map LangChain's provider-reported usage to Chat Completions usage."""
    if usage is None:
        return None
    return UsageInfo(
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
    )


def interrupt_tool_call(
    batch: LangGraphInterruptBatch,
    interrupt: Interrupt,
) -> ToolCall:
    """Format interrupt tool call."""
    return ToolCall(
        id=interrupt_tool_call_id(interrupt.id),
        type="function",
        function=ToolCallFunction(
            name=INTERRUPT_TOOL_NAME,
            arguments=interrupt_tool_arguments(batch, interrupt),
        ),
    )


def interrupt_tool_arguments(
    batch: LangGraphInterruptBatch,
    interrupt: Interrupt,
) -> str:
    """Format interrupt tool arguments."""
    return interrupt_arguments(
        run_id=batch.run_id,
        state_token=batch.state_token,
        payload=interrupt.value,
    )


class ChatCompletionStreamResponseBuilder:
    """Build OpenAI-compatible chat completion SSE chunks."""

    def __init__(self, model: str, *, include_usage: bool = False) -> None:
        self.response_id = f"chatcmpl-{uuid.uuid4()}"
        self.created = int(time.time())
        self.model = model
        self.include_usage = include_usage

    def role(self) -> str:
        """Stream role."""
        return self._chunk(ChatCompletionStreamResponseDelta(role=Role.ASSISTANT))

    def text(self, content: str) -> str:
        """Stream text content."""
        return self._chunk(ChatCompletionStreamResponseDelta(content=content))

    def client_event(self, extension: dict[str, object]) -> str:
        """Build an empty-delta chunk carrying the opt-in event extension."""
        return self._chunk(
            ChatCompletionStreamResponseDelta(),
            client_event_extension=extension,
        )

    def interrupt(self, batch: LangGraphInterruptBatch) -> str:
        """Stream interrupt."""
        return self._chunk(
            ChatCompletionStreamResponseDelta(
                tool_calls=[
                    ChatCompletionStreamToolCall(
                        index=index,
                        id=interrupt_tool_call_id(interrupt.id),
                        type="function",
                        function=ChatCompletionStreamToolCallFunction(
                            name=INTERRUPT_TOOL_NAME,
                            arguments=interrupt_tool_arguments(batch, interrupt),
                        ),
                    )
                    for index, interrupt in enumerate(batch.interrupts)
                ],
            ),
        )

    def tool_calls(self, message: AIMessage) -> str:
        """Stream complete final-message tool calls as one delta."""
        return self._chunk(
            ChatCompletionStreamResponseDelta(
                tool_calls=[
                    ChatCompletionStreamToolCall(
                        index=index,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=ChatCompletionStreamToolCallFunction(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for index, tool_call in enumerate(tool_calls_from_message(message))
                ]
            )
        )

    def finish(
        self,
        finish_reason: str,
        *,
        annotations: list[Annotation] | None = None,
    ) -> str:
        """Stream finish."""
        return self._chunk(
            ChatCompletionStreamResponseDelta(),
            finish_reason=finish_reason,
            annotations=annotations,
        )

    def error(self, message: str) -> str:
        """Stream error."""
        return self._format_data(
            openai_error_payload(ErrorObject(message=message, type="server_error"))
        )

    def done(self) -> str:  # ruff: ignore[no-self-use]
        """Stream done."""
        return "data: [DONE]\n\n"

    def usage(self, usage: UsageMetadata) -> str:
        """Stream the optional final usage-only chunk."""
        response = ChatCompletionStreamResponse(
            id=self.response_id,
            created=self.created,
            model=self.model,
            choices=[],
            usage=usage_info(usage),
        )
        return self._format_data(response.model_dump(mode="json", exclude_none=True))

    def _chunk(
        self,
        delta: ChatCompletionStreamResponseDelta,
        finish_reason: str | None = None,
        annotations: list[Annotation] | None = None,
        client_event_extension: dict[str, object] | None = None,
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
        # We apply exclude_none=True here because SSE (StreamingResponse) chunks
        # bypass FastAPI's route-level response_model_exclude_none setting.
        # This prevents sending bloated chunks and matches OpenAI's REST behavior.
        data = response.model_dump(mode="json", exclude_none=True)
        if self.include_usage:
            data["usage"] = None
        if annotations:
            # The Chat Completions delta schema omits annotations, so add the
            # compatibility extension after validating the standard chunk.
            data["choices"][0]["delta"]["annotations"] = [
                annotation.model_dump(mode="json", exclude_none=True)
                for annotation in annotations
            ]
        if client_event_extension is not None:
            # Event extensions remain complete Chat Completions chunks; their
            # empty delta keeps extension data separate from assistant text.
            data["choices"][0]["delta"] = {}
            data["langgraph_openai_serve"] = client_event_extension
        return self._format_data(data)

    def _format_data(self, data: dict) -> str:  # ruff: ignore[no-self-use]
        return f"data: {json.dumps(data)}\n\n"
