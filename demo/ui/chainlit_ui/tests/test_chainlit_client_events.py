import importlib
from unittest.mock import AsyncMock, MagicMock, Mock, call

import pytest
from openai.types.chat import ChatCompletionChunk


class Session:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


@pytest.mark.anyio
async def test_message_handler_renders_public_client_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simple = importlib.import_module("lgos_chainlit.simple")
    session = Session({"chat_profile": "custom-event-showcase"})
    messages = [{"role": "user", "content": "Build the report."}]

    def chunk(
        *,
        extension: dict[str, object] | None,
        content: str | None = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-demo",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "custom-event-showcase",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
                "langgraph_openai_serve": extension,
            }
        )

    public_events: list[dict[str, object]] = [
        {
            "schema_version": 1,
            "event": {
                "type": "status",
                "namespace": ["research"],
                "data": {"message": "Planning compatibility report"},
            },
        },
        {
            "schema_version": 1,
            "event": {
                "type": "progress",
                "namespace": ["research"],
                "data": {"completed": 2, "total": 5},
            },
        },
        {
            "schema_version": 1,
            "event": {
                "type": "artifact",
                "namespace": ["research", "report"],
                "data": {"id": "report-1"},
            },
        },
    ]
    chunks = [chunk(extension=event) for event in public_events]
    chunks.append(chunk(extension=None, content="Docs answer"))
    stream = MagicMock()
    stream.__aiter__.return_value = iter(chunks)
    stream.close = AsyncMock()
    create = AsyncMock(return_value=stream)
    assistant_message = Mock(content="")
    assistant_message.stream_token = AsyncMock()
    assistant_message.update = AsyncMock()
    event_message = Mock(metadata=None, send=AsyncMock())
    message_factory = Mock(side_effect=[assistant_message, event_message])
    element = Mock(props={})
    element_updates: list[dict[str, object]] = []

    async def record_element_update() -> None:
        element_updates.append(element.props)

    element.update = AsyncMock(side_effect=record_element_update)
    custom_element_factory = Mock(return_value=element)

    monkeypatch.setattr(simple.cl, "user_session", session)
    monkeypatch.setattr(simple.cl, "Message", message_factory)
    monkeypatch.setattr(simple.cl, "CustomElement", custom_element_factory)
    monkeypatch.setattr(simple, "text_only_chat_messages", lambda: messages)
    monkeypatch.setattr(simple, "authenticated_user_identifier", lambda: "demo-user")
    monkeypatch.setattr(simple.client.chat.completions, "create", create)

    await simple.on_message(Mock(content=messages[0]["content"]))

    create.assert_awaited_once_with(
        model="custom-event-showcase",
        messages=messages,
        stream=True,
        user="demo-user",
        metadata={"langgraph_stream_events": "v1"},
    )
    assert message_factory.call_args_list == [
        call(content=""),
        call(content="", elements=[element]),
    ]
    custom_element_factory.assert_called_once_with(
        name="ClientEventTimeline",
        props={"events": [public_events[0]["event"]]},
    )
    assert event_message.metadata == {"lgos_chainlit.exclude_from_model_context": True}
    event_message.send.assert_awaited_once_with()
    assert element_updates == [
        {"events": [event["event"] for event in public_events[:2]]},
        {"events": [event["event"] for event in public_events]},
    ]
    assert element.update.await_count == len(public_events) - 1
    assistant_message.stream_token.assert_awaited_once_with("Docs answer")
    assistant_message.update.assert_awaited_once_with()
    stream.close.assert_awaited_once_with()
