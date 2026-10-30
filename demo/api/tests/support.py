"""A fake OpenAI Responses provider for graphs that call models through LGOS."""

import itertools
import json
from collections import deque
from collections.abc import Sequence
from typing import Any

import httpx2

# Provider IDs are unique, and add_messages replaces a message with a known ID.
_ids = itertools.count()


def model_response(
    text: str = "",
    *,
    calls: Sequence[dict[str, Any]] = (),
    refusal: str | None = None,
) -> dict[str, Any]:
    """Build a completed Responses result."""
    number = next(_ids)
    content = (
        {"type": "refusal", "refusal": refusal}
        if refusal
        else {"type": "output_text", "text": text, "annotations": []}
    )
    return {
        "id": f"resp_{number}",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-luna",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            *calls,
            {
                "id": f"msg_{number}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [content],
            },
        ],
    }


def text_chunks(text: str) -> list[str]:
    """Split text into the two chunks a streamed provider call sends."""
    middle = len(text) // 2
    return [chunk for chunk in (text[:middle], text[middle:]) if chunk]


def response_events(response: dict[str, Any]) -> bytes:
    """Encode a Responses result as the SSE events of a streamed call."""
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        }
    ]
    for index, item in enumerate(response["output"]):
        if item["type"] == "function_call":
            added = {**item, "arguments": ""}
            deltas = [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "delta": item["arguments"],
                }
            ]
        elif item["type"] != "message":
            # Other items, such as custom tool calls, arrive whole.
            added, deltas = item, []
        elif (part := item["content"][0])["type"] == "refusal":
            added = {**item, "content": []}
            deltas = [
                {
                    "type": "response.refusal.done",
                    "item_id": item["id"],
                    "content_index": 0,
                    "refusal": part["refusal"],
                }
            ]
        else:
            added = {**item, "content": []}
            deltas = [
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "content_index": 0,
                    "delta": chunk,
                    "logprobs": [],
                }
                for chunk in text_chunks(part["text"])
            ]
        events += [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": added,
            },
            *({**delta, "output_index": index} for delta in deltas),
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]
    events.append({"type": "response.completed", "response": response})
    return "".join(
        f"event: {event['type']}\n"
        f"data: {json.dumps({**event, 'sequence_number': number})}\n\n"
        for number, event in enumerate(events)
    ).encode()


class ModelProvider:
    """Serve queued Responses results and record every request payload."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        # Like OpenAI, echo the text format; LangChain parses streamed
        # structured output only when the response declares its JSON schema.
        response = {**self.responses.popleft(), "text": payload.get("text")}
        if not payload.get("stream"):
            return httpx2.Response(200, json=response)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_events(response),
        )
