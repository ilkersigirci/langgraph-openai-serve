"""Chainlit HITL demo that communicates only through the OpenAI client."""

import json
import os
from typing import Any

import chainlit as cl
from openai import AsyncOpenAI

from langgraph_openai_serve.hitl.openai import HITL_TOOL_NAME

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "demo"),
    base_url=os.getenv("LANGGRAPH_OPENAI_BASE_URL", "http://localhost:8000/v1"),
)

HITL_TOOL = {
    "type": "function",
    "function": {
        "name": HITL_TOOL_NAME,
        "description": "Collect human input needed to resume a paused graph",
        "parameters": {"type": "object", "additionalProperties": True},
    },
}


async def collect_human_response(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Render the appropriate Chainlit control for an interrupt payload."""
    if value.get("kind") == "tool_approval" or "action_request" in value:
        return await collect_tool_approval(value)

    raise ValueError(f"Unsupported interrupt payload: {value}")


async def collect_tool_approval(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Ask the user what to do with a proposed tool call."""
    request = value["action_request"]
    config = value.get("config", {})
    args = request.get("args", {})
    content = (
        "### Approve tool call\n\n"
        f"Tool: `{request.get('action')}`\n\n"
        f"Arguments:\n```json\n{json.dumps(args, indent=2)}\n```"
    )
    actions = []
    if config.get("allow_accept", True):
        actions.append(
            cl.Action(name="accept", label="Accept", payload={"type": "accept"})
        )
    if config.get("allow_edit", True):
        actions.append(cl.Action(name="edit", label="Edit", payload={"type": "edit"}))
    if config.get("allow_respond", True):
        actions.append(
            cl.Action(name="response", label="Respond", payload={"type": "response"})
        )
    if config.get("allow_ignore", True):
        actions.append(
            cl.Action(name="ignore", label="Ignore", payload={"type": "ignore"})
        )

    answer = await cl.AskActionMessage(
        content=content,
        actions=actions,
        timeout=300,
    ).send()
    action = answer["payload"]["type"] if answer else "ignore"

    if action == "edit":
        edited = await cl.AskUserMessage(
            content="Edit the tool arguments as JSON:",
            timeout=300,
        ).send()
        try:
            args = json.loads(edited["output"]) if edited else args
        except json.JSONDecodeError:
            await cl.Message(content="Invalid JSON. Keeping the original args.").send()
        return [{"type": "edit", "args": args}]

    if action == "response":
        response = await cl.AskUserMessage(
            content="Respond directly instead of running the tool:",
            timeout=300,
        ).send()
        return [{"type": "response", "args": response["output"] if response else ""}]

    return [{"type": action}]


def _append_tool_call_delta(tool_calls: dict[int, dict[str, Any]], tool_call) -> None:
    index = tool_call.index or 0
    current = tool_calls.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if tool_call.id:
        current["id"] = tool_call.id
    if tool_call.type:
        current["type"] = tool_call.type
    if tool_call.function:
        if tool_call.function.name:
            current["function"]["name"] += tool_call.function.name
        if tool_call.function.arguments:
            current["function"]["arguments"] += tool_call.function.arguments


async def stream_assistant_response(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stream assistant text while preserving OpenAI tool-call messages."""
    stream = await client.chat.completions.create(
        model="hitl-demo",
        messages=messages,
        tools=[HITL_TOOL],
        stream=True,
    )
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    ui_message: cl.Message | None = None

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            content_parts.append(delta.content)
            if ui_message is None:
                ui_message = cl.Message(content="")
            await ui_message.stream_token(delta.content)

        for tool_call in delta.tool_calls or []:
            _append_tool_call_delta(tool_calls, tool_call)

    if ui_message is not None:
        await ui_message.send()

    if tool_calls:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
        }
    return {"role": "assistant", "content": "".join(content_parts)}


@cl.on_chat_start
def start_chat() -> None:
    cl.user_session.set("messages", [])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    messages: list[dict[str, Any]] = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})

    while True:
        assistant = await stream_assistant_response(messages)
        messages.append(assistant)

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            return

        for tool_call in tool_calls:
            function = tool_call["function"]
            if function["name"] != HITL_TOOL_NAME:
                raise RuntimeError(f"Unsupported tool: {function['name']}")

            payload = json.loads(function["arguments"])
            interrupts = payload["interrupts"]
            responses = {
                item["id"]: await collect_human_response(item["value"])
                for item in interrupts
            }
            resume = (
                next(iter(responses.values())) if len(responses) == 1 else responses
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(resume),
                }
            )
