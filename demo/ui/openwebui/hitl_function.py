"""
title: LangGraph HITL Approval Modal
author: langgraph-openai-serve
version: 0.1
"""

import json
import os
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletion,
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
            default=os.getenv(
                "LGOS_OPENWEBUI_OPENAI_BASE_URL",
                "http://lgos-demo-api:8000/v1",
            ),
            description="Base URL for the LangGraph OpenAI-compatible API.",
        )
        OPENAI_API_KEY: str = Field(
            default=os.getenv("LGOS_OPENAI_API_KEY", "DUMMY"),
            description="Bearer token sent to the LangGraph API.",
        )
        MODEL: str = Field(
            default="interruptible-approval",
            description="Interrupt-enabled demo model to call.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict[str, Any],
        __event_call__: Any = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> str:
        thread_id = self._thread_id(__metadata__ or {})
        messages = cast(list[ChatCompletionMessageParam], body.get("messages") or [])

        try:
            response = await self._chat(messages, thread_id)
            if not response.choices:
                return NO_CHOICES_MESSAGE
            assistant_message = response.choices[0].message

            tool_call = self._interrupt_tool_call(assistant_message)
            if tool_call is None:
                return str(assistant_message.content or "")

            if __event_call__ is None:
                return "Open WebUI approval modal is unavailable for this request."

            approval = await __event_call__(self._approval_event(tool_call))
            if isinstance(approval, dict) and approval.get("error"):
                return f"Open WebUI approval failed: {approval['error']}"

            decision = "approve" if approval is True else "reject"
            response = await self._chat(
                [
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
                ],
                thread_id,
            )
        except OpenAIError as exc:
            return f"Error calling LangGraph API: {exc}"

        return (
            NO_CHOICES_MESSAGE
            if not response.choices
            else str(response.choices[0].message.content or "")
        )

    async def _chat(
        self,
        messages: list[ChatCompletionMessageParam],
        thread_id: str,
    ) -> ChatCompletion:
        client = AsyncOpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
            timeout=30,
        )
        return await client.chat.completions.create(
            model=self.valves.MODEL,
            messages=messages,
            stream=False,
            metadata={"langgraph_thread_id": thread_id},
        )

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
