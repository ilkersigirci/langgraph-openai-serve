"""
title: LangGraph OpenAI Pipe
author: langgraph-openai-serve
version: 0.4
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.lib.streaming.chat import AsyncChatCompletionStream, ContentDeltaEvent
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from pydantic import BaseModel, Field

INTERRUPT_TOOL_NAME = "langgraph_interrupt"
NO_CHOICES_MESSAGE = "LangGraph API returned no choices."


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="http://lgos-demo-api:8000/v1",
            description="Base URL for the LangGraph OpenAI-compatible API.",
        )
        OPENAI_API_KEY: str = Field(
            default="DUMMY",
            description="Bearer token sent to the LangGraph API.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipes(self) -> list[dict[str, str]]:
        """Expose every registered LangGraph model in Open WebUI's selector."""
        async with self._client() as client:
            models = await client.models.list()

        return [{"id": model.id, "name": model.id} for model in models.data]

    async def pipe(
        self,
        body: dict[str, Any],
        __event_call__: Any = None,
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        thread_id = self._thread_id(__metadata__ or {})
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])

        try:
            model_id = self._model_id(body)
        except ValueError as exc:
            yield str(exc)
            return

        try:
            async with self._chat(messages, thread_id, model_id) as stream:
                async for delta in self._content_deltas(stream):
                    yield delta
                response = await stream.get_final_completion()

            if not response.choices:
                yield NO_CHOICES_MESSAGE
                return

            assistant_message = response.choices[0].message
            await self._emit_citations(assistant_message, __event_emitter__)

            tool_call = self._interrupt_tool_call(assistant_message)
            if tool_call is None:
                return

            if __event_call__ is None:
                yield "Open WebUI approval modal is unavailable for this request."
                return

            approval = await __event_call__(self._approval_event(tool_call))
            if isinstance(approval, dict) and approval.get("error"):
                yield f"Open WebUI approval failed: {approval['error']}"
                return

            decision = "approve" if approval is True else "reject"
            resume_messages = [
                *messages,
                ChatCompletionAssistantMessageParam(
                    role=assistant_message.role,
                    content=assistant_message.content,
                    tool_calls=[
                        cast(
                            ChatCompletionMessageToolCallParam,
                            tool_call.model_dump(mode="json"),
                        )
                    ],
                ),
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=json.dumps({"resume": decision}),
                ),
            ]
            async with self._chat(resume_messages, thread_id, model_id) as stream:
                async for delta in self._content_deltas(stream):
                    yield delta
                response = await stream.get_final_completion()
        except OpenAIError as exc:
            yield f"Error calling LangGraph API: {exc}"
            return

        if not response.choices:
            yield NO_CHOICES_MESSAGE
        else:
            await self._emit_citations(
                response.choices[0].message,
                __event_emitter__,
            )

    async def _emit_citations(
        self,
        message: ChatCompletionMessage,
        event_emitter: Any,
    ) -> None:
        """Translate OpenAI URL annotations to native Open WebUI sources."""
        if event_emitter is None:
            return

        for annotation in message.annotations or []:
            citation = annotation.url_citation
            await event_emitter(
                {
                    "type": "source",
                    "data": {
                        "source": {
                            "name": citation.title,
                            "url": citation.url,
                        },
                        "document": [citation.title],
                        "metadata": [
                            {
                                "source": citation.url,
                                "name": citation.title,
                            }
                        ],
                    },
                }
            )

    async def _content_deltas(
        self,
        stream: AsyncChatCompletionStream[Any],
    ) -> AsyncIterator[str]:
        """Yield text while retaining internal tool calls in the SDK accumulator."""
        async for event in stream:
            if isinstance(event, ContentDeltaEvent):
                yield event.delta

    @asynccontextmanager
    async def _chat(
        self,
        messages: list[ChatCompletionMessageParam],
        thread_id: str,
        model_id: str,
    ) -> AsyncIterator[AsyncChatCompletionStream[Any]]:
        async with (
            self._client() as client,
            client.chat.completions.stream(
                model=model_id,
                messages=messages,
                metadata={"langgraph_thread_id": thread_id},
            ) as stream,
        ):
            yield stream

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=30,
        )

    def _model_id(self, body: dict[str, Any]) -> str:
        qualified_model_id = body.get("model")
        if not isinstance(qualified_model_id, str):
            raise ValueError("Open WebUI did not provide a valid model ID.")

        _, separator, model_id = qualified_model_id.partition(".")
        if not separator or not model_id:
            raise ValueError("Open WebUI did not provide a valid model ID.")

        return model_id

    def _thread_id(self, metadata: dict[str, Any]) -> str:
        value = metadata.get("chat_id") or metadata.get("session_id") or "default"
        return f"openwebui:function:{value}"

    def _interrupt_tool_call(
        self,
        message: ChatCompletionMessage,
    ) -> ChatCompletionMessageToolCall | None:
        for tool_call in message.tool_calls or []:
            if (
                isinstance(tool_call, ChatCompletionMessageToolCall)
                and tool_call.function.name == INTERRUPT_TOOL_NAME
            ):
                return tool_call
        return None

    def _approval_event(
        self,
        tool_call: ChatCompletionMessageToolCall,
    ) -> dict[str, Any]:
        payload = self._interrupt_payload(tool_call) or {}
        question = str(payload.get("question") or "Approve this agent action?")
        request = str(payload.get("request") or json.dumps(payload, indent=2))
        return {
            "type": "confirmation",
            "data": {
                "title": question,
                "message": request,
            },
        }

    def _interrupt_payload(
        self,
        tool_call: ChatCompletionMessageToolCall,
    ) -> dict[str, object] | None:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except (TypeError, json.JSONDecodeError):
            return None

        if not isinstance(arguments, dict):
            return None

        payload = arguments.get("payload")
        return payload if isinstance(payload, dict) else None
