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

from langgraph_openai_serve.api.tools import FunctionCall, encode_function_calls
from langgraph_openai_serve.core.errors import openai_error_payload
from langgraph_openai_serve.graph.citations import citations_from_message


def chat_completion_response(
    *,
    model: str,
    message: AIMessage,
) -> ChatCompletion:
    """Build a non-streaming OpenAI-compatible chat completion response."""
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=call.call_id,
            type="function",
            function=Function(name=call.name, arguments=call.arguments),
        )
        for call in encode_function_calls(message)
    ]
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=message.text or None,
                    annotations=annotations_from_message(message) or None,
                    tool_calls=tool_calls or None,
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=usage_info(message.usage_metadata),
    )


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
        """Encode the opening assistant-role chunk."""
        return self._chunk(ChoiceDelta(role="assistant"))

    def text(self, content: str) -> str:
        """Encode one text-delta chunk."""
        return self._chunk(ChoiceDelta(content=content))

    def tool_calls(self, calls: list[FunctionCall]) -> str:
        """Stream complete final-message tool calls as one delta."""
        return self._chunk(
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=index,
                        id=call.call_id,
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=call.name, arguments=call.arguments
                        ),
                    )
                    for index, call in enumerate(calls)
                ]
            )
        )

    def finish(
        self,
        finish_reason: Literal["stop", "tool_calls"],
        *,
        annotations: list[Annotation] | None = None,
    ) -> str:
        """Encode the finish chunk, with any citations on its delta."""
        return self._chunk(
            ChoiceDelta(),
            finish_reason=finish_reason,
            annotations=annotations,
        )

    def error(self, message: str) -> str:
        """Encode an OpenAI error payload as one data frame."""
        return self._format_data(
            openai_error_payload(ErrorObject(message=message, type="server_error"))
        )

    @staticmethod
    def done() -> str:
        """Encode the terminating [DONE] frame."""
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
        # OpenAI sends "usage": null on every chunk but the final usage chunk.
        if self.include_usage:
            data["usage"] = None
        # ChoiceDelta declares no annotations; LGOS adds them as an extension.
        if annotations:
            data["choices"][0]["delta"]["annotations"] = [
                annotation.model_dump(mode="json", exclude_none=True)
                for annotation in annotations
            ]
        return self._format_data(data)

    @staticmethod
    def _format_data(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"
