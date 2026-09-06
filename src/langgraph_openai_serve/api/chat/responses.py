"""OpenAI chat response builders."""

import json
import time
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, UsageMetadata
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import Annotation
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.shared import ErrorObject

from langgraph_openai_serve.core.errors import openai_error_payload
from langgraph_openai_serve.graph.citations import citations_from_message


def chat_completion_response(
    *,
    model: str,
    message: AIMessage,
) -> ChatCompletion:
    """Build a non-streaming OpenAI-compatible chat completion response."""
    resp_message, finish_reason = response_message(message)
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=resp_message,
                finish_reason=finish_reason,
            )
        ],
        usage=usage_info(message.usage_metadata),
    )


def response_message(
    message: AIMessage,
) -> tuple[ChatCompletionMessage, Literal["stop", "tool_calls"]]:
    """Format response message."""
    tool_calls = tool_calls_from_message(message)
    return (
        ChatCompletionMessage(
            role="assistant",
            content=message.text or None,
            annotations=annotations_from_message(message) or None,
            tool_calls=tool_calls or None,
        ),
        "tool_calls" if tool_calls else "stop",
    )


def tool_calls_from_message(
    message: AIMessage,
) -> list[ChatCompletionMessageFunctionToolCall]:
    """Convert native LangChain tool calls to Chat Completions tool calls."""
    tool_calls = []
    for tool_call in message.tool_calls:
        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            msg = "Final AIMessage tool calls must have an id."
            raise ValueError(msg)
        tool_calls.append(
            ChatCompletionMessageFunctionToolCall(
                id=tool_call_id,
                type="function",
                function=Function(
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


def usage_info(usage: UsageMetadata | None) -> CompletionUsage | None:
    """Map LangChain's provider-reported usage to Chat Completions usage."""
    if usage is None:
        return None
    return CompletionUsage(
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
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
        return self._chunk(ChoiceDelta(role="assistant"))

    def text(self, content: str) -> str:
        """Stream text content."""
        return self._chunk(ChoiceDelta(content=content))

    def tool_calls(self, message: AIMessage) -> str:
        """Stream complete final-message tool calls as one delta."""
        return self._chunk(
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=index,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=ChoiceDeltaToolCallFunction(
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
        finish_reason: Literal["stop", "tool_calls"],
        *,
        annotations: list[Annotation] | None = None,
    ) -> str:
        """Stream finish."""
        return self._chunk(
            ChoiceDelta(),
            finish_reason=finish_reason,
            annotations=annotations,
        )

    def error(self, message: str) -> str:
        """Stream error."""
        return self._format_data(
            openai_error_payload(ErrorObject(message=message, type="server_error"))
        )

    @staticmethod
    def done() -> str:
        """Stream done."""
        return "data: [DONE]\n\n"

    def usage(self, usage: UsageMetadata) -> str:
        """Stream the optional final usage-only chunk."""
        response = ChatCompletionChunk(
            id=self.response_id,
            object="chat.completion.chunk",
            created=self.created,
            model=self.model,
            choices=[],
            usage=usage_info(usage),
        )
        return self._format_data(response.model_dump(mode="json", exclude_none=True))

    def _chunk(
        self,
        delta: ChoiceDelta,
        finish_reason: Literal["stop", "tool_calls"] | None = None,
        annotations: list[Annotation] | None = None,
    ) -> str:
        response = ChatCompletionChunk(
            id=self.response_id,
            object="chat.completion.chunk",
            created=self.created,
            model=self.model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )
        data = response.model_dump(mode="json", exclude_none=True)
        if self.include_usage:
            data["usage"] = None
        if annotations:
            data["choices"][0]["delta"]["annotations"] = [
                annotation.model_dump(mode="json", exclude_none=True)
                for annotation in annotations
            ]
        return self._format_data(data)

    @staticmethod
    def _format_data(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"


__all__ = [
    "ChatCompletionStreamResponseBuilder",
    "annotations_from_message",
    "chat_completion_response",
    "response_message",
    "tool_calls_from_message",
    "usage_info",
]
