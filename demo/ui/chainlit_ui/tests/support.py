"""Builders for OpenAI gateway replies and Chainlit conversation state."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import chainlit as cl
import httpx2
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import Annotation

from lgos_chainlit.chat_settings import configure_chat_settings

Reply = httpx2.Response | Callable[[httpx2.Request], httpx2.Response]


@dataclass
class FakeGateway:
    """Answer OpenAI requests in order and record what the client sent."""

    replies: list[Reply] = field(default_factory=list)
    requests: list[httpx2.Request] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        self.requests.append(request)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, httpx2.Response) else reply(request)

    def bodies(self, path: str) -> list[dict[str, Any]]:
        """Return the JSON bodies posted to one route."""
        return [
            json.loads(request.content)
            for request in self.requests
            if request.method == "POST" and request.url.path == path
        ]


def response(
    *output: ResponseOutputItem,
    id: str = "resp_test",
    status: str = "completed",
    **fields: Any,
) -> Response:
    return Response.model_validate(
        {
            "id": id,
            "object": "response",
            "created_at": 0,
            "model": "lgos-a/test",
            "status": status,
            "output": list(output),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            **fields,
        }
    )


def message(
    text: str,
    *,
    id: str = "msg_answer",
    phase: str | None = "final_answer",
    annotations: Sequence[Annotation] = (),
) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=id,
        type="message",
        role="assistant",
        status="completed",
        phase=phase,
        content=[
            ResponseOutputText(
                type="output_text", text=text, annotations=list(annotations)
            )
        ],
    )


def function_call(
    name: str, arguments: str, *, call_id: str
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}",
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
        type="function_call",
    )


def reply(response: Response) -> httpx2.Response:
    return httpx2.Response(200, json=response.model_dump(mode="json"))


def streamed(response: Response, *, deltas: bool = True) -> httpx2.Response:
    """Stream a Response as text deltas followed by its terminal event.

    Without deltas, the stream resembles a proxy that forwards only lifecycle
    events.
    """
    payload = response.model_dump(mode="json")
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {**payload, "status": "in_progress", "output": []},
        }
    ]
    for output_index, item in enumerate(payload["output"] if deltas else []):
        is_message = item["type"] == "message"
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "content": []} if is_message else item,
            }
        )
        for content_index, part in enumerate(item["content"] if is_message else []):
            location = {
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": content_index,
            }
            if part["type"] == "refusal":
                events += [
                    {
                        "type": "response.content_part.added",
                        **location,
                        "part": {**part, "refusal": ""},
                    },
                    {
                        "type": "response.refusal.delta",
                        **location,
                        "delta": part["refusal"],
                    },
                ]
                continue
            events += [
                {
                    "type": "response.content_part.added",
                    **location,
                    "part": {**part, "text": ""},
                },
                {
                    "type": "response.output_text.delta",
                    **location,
                    "delta": part["text"],
                    "logprobs": [],
                },
                {
                    "type": "response.output_text.done",
                    **location,
                    "text": part["text"],
                    "logprobs": [],
                },
            ]
    events.append({"type": f"response.{response.status}", "response": payload})
    return sse(*events)


def unfinished_answer(*deltas: str) -> list[dict[str, Any]]:
    """Start one answer and stream these text deltas without finishing it."""
    payload = response(message("".join(deltas))).model_dump(mode="json")
    item = payload["output"][0]
    part = {"item_id": item["id"], "output_index": 0, "content_index": 0}
    return [
        {
            "type": "response.created",
            "response": {**payload, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "content": []},
        },
        {
            "type": "response.content_part.added",
            **part,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        *(
            {
                "type": "response.output_text.delta",
                **part,
                "delta": delta,
                "logprobs": [],
            }
            for delta in deltas
        ),
    ]


def sse(*events: dict[str, Any]) -> httpx2.Response:
    """Reply with these Responses stream events."""
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(
            f"event: {event['type']}\n"
            f"data: {json.dumps({**event, 'sequence_number': number})}\n\n"
            for number, event in enumerate(events)
        ),
    )


def model_info(models: dict[str, dict[str, object]]) -> httpx2.Response:
    """Reply to LiteLLM's /model/info with each model's LGOS extension."""
    return httpx2.Response(
        200,
        json={
            "data": [
                {"model_name": name, "model_info": {"lgos": lgos}}
                for name, lgos in models.items()
            ]
        },
    )


async def select_profile(
    gateway: FakeGateway,
    model_id: str,
    **lgos: object,
) -> None:
    """Select a chat profile the way Chainlit does, then load its settings."""
    cl.context.session.chat_profile = model_id
    gateway.replies.append(
        model_info({model_id: {"description": "Demo graph", "features": [], **lgos}})
    )
    await configure_chat_settings()


def user_message(content: str, **fields: Any) -> cl.Message:
    """Add a user turn to the chat context, as Chainlit does before on_message."""
    return cl.chat_context.add(
        cl.Message(content=content, type="user_message", **fields)
    )


def transcript() -> list[str]:
    return [message.content for message in cl.chat_context.get()]
