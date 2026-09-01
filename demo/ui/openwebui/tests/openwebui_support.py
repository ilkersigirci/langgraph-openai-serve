import json
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

USER_REQUEST = "Refund order ORDER-123"
UPSTREAM_MODEL_ID = "interruptible-approval"
MODEL_ID = f"lgos-a/{UPSTREAM_MODEL_ID}"
RUN_ID = "725c277a-f6d5-4c52-95eb-8c09e91f7a7c"
STATE_TOKEN = "state-token-1"
MARKDOWN_DELTAS = (
    "Read the [source](https://example.com/source) [1], ",
    "view ![diagram](https://example.com/diagram.png), ",
    "and follow the [audio link](https://example.com/overview.mp3).",
)
MARKDOWN_RESPONSE = "".join(MARKDOWN_DELTAS)
INTERRUPT_PAYLOAD = {
    "action": "refund",
    "question": "How should the refund be handled?",
    "request": USER_REQUEST,
    "choices": ["approve", "reject"],
    "allow_other": True,
}


class ScriptedStream:
    def __init__(
        self,
        events: Sequence[str | ChatCompletionChunk],
        completion: ChatCompletion,
    ) -> None:
        self._events = events
        self._completion = completion

    async def __aenter__(self) -> "ScriptedStream":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        for event in self._events:
            if isinstance(event, str):
                yield stream_chunk(content=event)
            else:
                yield event

        message = self._completion.choices[0].message
        if message.tool_calls:
            yield stream_chunk(
                tool_calls=[
                    {
                        "index": index,
                        **tool_call.model_dump(mode="json", exclude_none=True),
                    }
                    for index, tool_call in enumerate(message.tool_calls)
                ]
            )
            yield stream_chunk(finish_reason="tool_calls")
        elif message.annotations:
            yield stream_chunk(
                finish_reason="stop",
                annotations=[
                    annotation.model_dump(mode="json", exclude_none=True)
                    for annotation in message.annotations
                ],
            )


class ScriptedChat:
    def __init__(
        self,
        *steps: tuple[Sequence[str | ChatCompletionChunk], ChatCompletion],
    ) -> None:
        self._steps = steps
        self.calls: list[tuple[list[dict[str, Any]], str]] = []

    @asynccontextmanager
    async def __call__(
        self,
        *,
        client: Any,
        messages: list[dict[str, Any]],
        model_id: str,
        request_metadata: dict[str, str] | None = None,
        include_client_events: bool = False,
        user_id: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[ScriptedStream]:
        step_index = len(self.calls)
        self.calls.append((messages, model_id))
        if step_index >= len(self._steps):
            msg = f"Unexpected chat call {step_index + 1}"
            raise AssertionError(msg)

        deltas, completion = self._steps[step_index]
        yield ScriptedStream(deltas, completion)


def stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if annotations is not None:
        delta["annotations"] = annotations
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": UPSTREAM_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


async def collect_response(
    pipe_response: Awaitable[
        AsyncIterator[str | dict[str, Any]] | str | dict[str, Any]
    ],
) -> list[str | dict[str, Any]]:
    response = await pipe_response
    if isinstance(response, str | dict):
        return [response]
    return [chunk async for chunk in response]


def body(
    content: str,
    model: str = f"generic.{MODEL_ID}",
    *,
    stream: bool = True,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }


def completion(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if annotations is not None:
        message["annotations"] = annotations
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": UPSTREAM_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
        }
    )


def model(
    *,
    features: list[str] | None = None,
    client_settings: dict[str, Any] | None = None,
) -> SimpleNamespace:
    extension: dict[str, Any] = {
        "schema_version": 1,
        "description": "DUMMY",
        "features": features or [],
    }
    if client_settings is not None:
        extension["client_settings"] = client_settings
    return SimpleNamespace(
        model_extra={
            "langgraph_openai_serve": extension,
        }
    )


def interrupt_call(
    interrupt_id: str,
    payload: object,
    *,
    arguments: object | None = None,
    state_token: str = STATE_TOKEN,
) -> dict[str, Any]:
    arguments = (
        {
            "run_id": RUN_ID,
            "state_token": state_token,
            "payload": payload,
        }
        if arguments is None
        else arguments
    )
    return {
        "id": f"lg_interrupt_{interrupt_id}",
        "type": "function",
        "function": {
            "name": "langgraph_interrupt",
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }


def interrupt_response(arguments: object | None = None) -> ChatCompletion:
    return completion(
        tool_calls=[
            interrupt_call(
                "interrupt-1",
                INTERRUPT_PAYLOAD,
                arguments=arguments,
            )
        ]
    )


def citation_response() -> ChatCompletion:
    citation_text = "source"
    start = MARKDOWN_RESPONSE.index(citation_text)
    return completion(
        MARKDOWN_RESPONSE,
        annotations=[
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": start,
                    "end_index": start + len(citation_text) - 1,
                    "title": "Example source",
                    "url": "https://example.com/source",
                },
            }
        ],
    )
