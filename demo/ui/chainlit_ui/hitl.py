"""Chainlit UI for the LangGraph interrupt demo graph."""

import json
import os
from typing import cast

import chainlit as cl
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from langgraph_openai_serve.api.chat.utils.interrupts import INTERRUPT_TOOL_NAME

OPENAI_BASE_URL = os.getenv("LGOS_CHAINLIT_OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("LGOS_OPENAI_API_KEY", "DUMMY")
HITL_MODEL = os.getenv("LGOS_CHAINLIT_HITL_MODEL", "interruptible-approval")

client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


@cl.set_chat_profiles
async def set_chat_profiles() -> list[cl.ChatProfile]:
    models = await client.models.list()
    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description="Approve or reject a LangGraph interrupt.",
        )
        for model in models.data
        if model.id == HITL_MODEL
    ]


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Approval",
            message="Refund order ORDER-123 for the customer.",
        )
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("messages", [])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    messages = chat_messages()
    messages.append(
        ChatCompletionUserMessageParam(role="user", content=message.content)
    )

    response = await create_completion(messages)
    assistant_message = response.choices[0].message
    tool_call = interrupt_tool_call(assistant_message)

    if tool_call is not None:
        messages.append(assistant_tool_call_message(assistant_message, tool_call))
        resume_value = await ask_for_resume(tool_call)
        if resume_value is None:
            return

        messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=json.dumps({"resume": resume_value}),
            )
        )
        response = await create_completion(messages)
        assistant_message = response.choices[0].message

    content = assistant_message.content or ""
    messages.append(
        ChatCompletionAssistantMessageParam(
            role=assistant_message.role,
            content=content,
        )
    )
    cl.user_session.set("messages", messages)
    await cl.Message(content=content).send()


def chat_messages() -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], cl.user_session.get("messages") or [])


async def create_completion(
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletion:
    return await client.chat.completions.create(
        model=cl.user_session.get("chat_profile") or HITL_MODEL,
        messages=messages,
        metadata={"langgraph_thread_id": cl.context.session.thread_id},
    )


def assistant_tool_call_message(
    message: ChatCompletionMessage,
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionAssistantMessageParam:
    return ChatCompletionAssistantMessageParam(
        role=message.role,
        content=message.content,
        tool_calls=[tool_call_param(tool_call)],
    )


def tool_call_param(
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionMessageToolCallParam:
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type=tool_call.type,
        function={
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    )


async def ask_for_resume(tool_call: ChatCompletionMessageToolCall) -> str | None:
    payload = interrupt_payload(tool_call)
    if payload is None:
        await cl.Message(content="Received an unsupported interrupt payload.").send()
        return None

    response = await cl.AskActionMessage(
        content=interrupt_prompt(payload),
        actions=[
            cl.Action(
                name="approve",
                label="Approve",
                icon="check",
                payload={"resume": "approve"},
            ),
            cl.Action(
                name="reject",
                label="Reject",
                icon="x",
                payload={"resume": "reject"},
            ),
        ],
        timeout=300,
    ).send()

    if not response:
        await cl.Message(content="Approval timed out.").send()
        return None

    payload = response.get("payload") or {}
    return str(payload.get("resume") or "reject")


def interrupt_tool_call(
    message: ChatCompletionMessage,
) -> ChatCompletionMessageToolCall | None:
    for tool_call in message.tool_calls or []:
        if (
            isinstance(tool_call, ChatCompletionMessageToolCall)
            and tool_call.function.name == INTERRUPT_TOOL_NAME
        ):
            return tool_call
    return None


def interrupt_payload(
    tool_call: ChatCompletionMessageToolCall,
) -> dict[str, object] | None:
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        return None

    payload = arguments.get("payload")
    return payload if isinstance(payload, dict) else None


def interrupt_prompt(payload: dict[str, object]) -> str:
    lines = [str(payload.get("question") or "Approve this action?")]
    if payload.get("request"):
        lines.append(f"Request: {payload['request']}")

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        lines.append("Choices: " + ", ".join(str(choice) for choice in choices))

    return "\n\n".join(lines)
