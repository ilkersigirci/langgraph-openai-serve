"""Chainlit demo for OpenAI-compatible UI events.

Run the API server separately, for example:

    uv run --module demo.app

Then run this UI with:

    make run-ui-chainlit
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import chainlit as cl
import httpx

BASE_URL = os.getenv("LG_OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("LG_OPENAI_API_KEY", "test")
MODEL = os.getenv("LG_OPENAI_MODEL", "simple-graph-no-history")
REQUEST_TIMEOUT = float(os.getenv("LG_OPENAI_TIMEOUT", "120"))
STREAM_EVENTS = os.getenv("LG_OPENAI_STREAM", "true").lower() not in {
    "0",
    "false",
    "no",
}


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("messages", [])
    cl.user_session.set("thread_id", f"chainlit-{uuid.uuid4()}")
    await cl.Message(
        content=f"Connected to `{BASE_URL}` using model `{MODEL}`.",
        author="system",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    messages = cl.user_session.get("messages") or []
    messages.append({"role": "user", "content": message.content})
    cl.user_session.set("messages", messages)

    if STREAM_EVENTS:
        response = await _stream_event_completion(messages)
    else:
        response = await _non_stream_event_completion(messages)

    if response.get("tool_calls"):
        await _handle_ui_event_tool_call(response["tool_calls"][0])
    elif response.get("content"):
        messages.append({"role": "assistant", "content": response["content"]})
        cl.user_session.set("messages", messages)


async def _stream_event_completion(messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "x_langgraph_openai_serve": {
            "ui_events": {
                "enabled": True,
                "thread_id": cl.user_session.get("thread_id"),
            }
        },
    }

    assistant_message = cl.Message(content="", author="assistant")
    await assistant_message.send()
    event_status = cl.Message(content="", author="ui-events")
    await event_status.send()

    raw_content = ""
    semantic_text = ""
    seen_events: list[str] = []

    async with (
        httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client,
        client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            raw_delta = delta.get("content")
            if not raw_delta:
                continue

            raw_content += raw_delta
            for event in _parse_event_lines(raw_delta):
                event_type = event.get("type", "unknown")
                seen_events.append(event_type)

                if event_type == "TEXT_MESSAGE_CONTENT":
                    text = event.get("delta", "")
                    semantic_text += text
                    await assistant_message.stream_token(text)
                elif event_type in {
                    "RUN_ERROR",
                    "CUSTOM",
                    "STATE_SNAPSHOT",
                    "STATE_DELTA",
                }:
                    await _send_event_card(event)

                event_status.content = " -> ".join(seen_events[-8:])
                await event_status.update()

    if not semantic_text and raw_content:
        assistant_message.content = raw_content
    await assistant_message.update()
    return {"content": raw_content, "tool_calls": None}


async def _handle_ui_event_tool_call(tool_call: dict[str, Any]) -> None:
    event = json.loads(tool_call["function"]["arguments"])
    await _send_event_card(event)

    action = await cl.AskActionMessage(
        content="The graph requested a UI response.",
        actions=[
            cl.Action(
                name="approve",
                label="Approve",
                payload={"approved": True},
            ),
            cl.Action(
                name="reject",
                label="Reject",
                payload={"approved": False},
            ),
        ],
        timeout=300,
    ).send()
    payload = getattr(action, "payload", None)
    if payload is None and isinstance(action, dict):
        payload = action.get("payload")
    payload = payload or {"approved": False}

    messages = cl.user_session.get("messages") or []
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(payload),
        }
    )
    cl.user_session.set("messages", messages)

    resumed = await _non_stream_event_completion(messages)
    if resumed.get("tool_calls"):
        await _handle_ui_event_tool_call(resumed["tool_calls"][0])
    else:
        cl.user_session.set("messages", [*messages, resumed])


async def _non_stream_event_completion(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "x_langgraph_openai_serve": {
            "ui_events": {
                "enabled": True,
                "thread_id": cl.user_session.get("thread_id"),
            }
        },
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        )
        response.raise_for_status()

    message = response.json()["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return message

    raw_content = message.get("content") or ""
    rendered = cl.Message(content="", author="assistant")
    await rendered.send()
    for event in _parse_event_lines(raw_content):
        if event.get("type") == "TEXT_MESSAGE_CONTENT":
            await rendered.stream_token(event.get("delta", ""))
        elif event.get("type") in {
            "RUN_ERROR",
            "CUSTOM",
            "STATE_SNAPSHOT",
            "STATE_DELTA",
        }:
            await _send_event_card(event)
    await rendered.update()
    return message


def _parse_event_lines(content: str) -> list[dict[str, Any]]:
    events = []
    for line in content.splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


async def _send_event_card(event: dict[str, Any]) -> None:
    await cl.Message(
        content=f"```json\n{json.dumps(event, indent=2)}\n```",
        author=event.get("type", "ui-event"),
    ).send()
