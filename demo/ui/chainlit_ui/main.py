"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import json
import os
from typing import Any

import chainlit as cl
from openai import AsyncOpenAI

OPENAI_BASE_URL = os.getenv("LGOS_CHAINLIT_OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("LGOS_OPENAI_API_KEY", "DUMMY")

client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
INTERRUPT_TOOL_NAME = "langgraph_interrupt"


@cl.set_chat_profiles
async def set_chat_profiles() -> list[cl.ChatProfile]:
    models = await client.models.list()

    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description=f"Talk to `{model.id}` from the demo backend.",
        )
        for model in models.data
    ]


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="About",
            message="Tell me about yourself.",
            icon="",
        ),
        cl.Starter(
            label="History",
            message="Remember that my favorite color is green.",
            icon="",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("messages", [])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    model = cl.user_session.get("chat_profile")
    messages: list[dict[str, Any]] = cl.user_session.get("messages") or []
    messages.append({"role": "user", "content": message.content})

    await continue_graph(model, messages)


async def continue_graph(model: str, messages: list[dict[str, Any]]) -> None:
    assistant_content, tool_call = await stream_assistant_response(model, messages)
    if tool_call is None:
        messages.append({"role": "assistant", "content": assistant_content})
        cl.user_session.set("messages", messages)
        return

    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
    cl.user_session.set("messages", messages)
    await resume_interrupt(model, messages, tool_call)


async def stream_assistant_response(
    model: str,
    messages: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    assistant_message = cl.Message(content="")
    interrupt_tool_call: dict[str, Any] | None = None

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        metadata={"langgraph_thread_id": cl.context.session.thread_id},
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        token = delta.content or ""
        if token:
            await assistant_message.stream_token(token)

        tool_call = (delta.tool_calls or [None])[0]
        if (
            tool_call
            and tool_call.function
            and tool_call.function.name == INTERRUPT_TOOL_NAME
        ):
            interrupt_tool_call = tool_call.model_dump(exclude_none=True)
            interrupt_tool_call.pop("index", None)

    if assistant_message.content:
        await assistant_message.update()
    return assistant_message.content, interrupt_tool_call


async def resume_interrupt(
    model: str,
    messages: list[dict[str, Any]],
    tool_call: dict[str, Any],
) -> None:
    response = await cl.AskUserMessage(
        content=interrupt_prompt(interrupt_payload(tool_call)),
        timeout=300,
    ).send()
    resume_value = str((response or {}).get("output") or "")

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps({"resume": resume_value}),
        }
    )

    await continue_graph(model, messages)


def interrupt_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}

    payload = arguments.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def interrupt_prompt(payload: dict[str, Any]) -> str:
    question = payload.get("question") or "Human input required."
    request = payload.get("request")

    if request:
        return f"{question}\n\nRequest:\n{request}"
    return str(question)
