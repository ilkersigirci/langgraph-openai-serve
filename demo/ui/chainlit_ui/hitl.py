"""Chainlit UI for the LangGraph interrupt demo graph."""

import json
import os
from typing import Any

import chainlit as cl
from openai import AsyncOpenAI

OPENAI_BASE_URL = os.getenv("LGOS_CHAINLIT_OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("LGOS_OPENAI_API_KEY", "DUMMY")
HITL_MODEL = os.getenv("LGOS_CHAINLIT_HITL_MODEL", "interruptible-approval")

INTERRUPT_TOOL_NAME = "langgraph_interrupt"

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
    messages: list[dict[str, Any]] = cl.user_session.get("messages") or []
    messages.append({"role": "user", "content": message.content})

    response = await create_completion(messages)
    message = response.choices[0].message
    message_data = message.model_dump(exclude_none=True)
    tool_call = interrupt_tool_call(message_data)

    if tool_call is not None:
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        )
        resume_value = await ask_for_resume(tool_call)
        if resume_value is None:
            return

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps({"resume": resume_value}),
            }
        )
        response = await create_completion(messages)
        message = response.choices[0].message

    content = message.content or ""
    messages.append({"role": "assistant", "content": content})
    cl.user_session.set("messages", messages)
    await cl.Message(content=content).send()


async def create_completion(messages: list[dict[str, Any]]) -> Any:
    return await client.chat.completions.create(
        model=cl.user_session.get("chat_profile") or HITL_MODEL,
        messages=messages,
        metadata={"langgraph_thread_id": cl.context.session.thread_id},
    )


async def ask_for_resume(tool_call: dict[str, Any]) -> str | None:
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


def interrupt_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        if function.get("name") == INTERRUPT_TOOL_NAME:
            return tool_call
    return None


def interrupt_payload(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None

    payload = arguments.get("payload")
    return payload if isinstance(payload, dict) else None


def interrupt_prompt(payload: dict[str, Any]) -> str:
    lines = [str(payload.get("question") or "Approve this action?")]
    if payload.get("request"):
        lines.append(f"Request: {payload['request']}")

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        lines.append("Choices: " + ", ".join(str(choice) for choice in choices))

    return "\n\n".join(lines)
