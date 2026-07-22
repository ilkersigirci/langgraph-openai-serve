"""Render LGOS client events in Chainlit."""

import chainlit as cl
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from lgos_chainlit.lgos_protocol import LGOS_EXTENSION_KEY, ClientEventExtension
from lgos_chainlit.utils.chat import mark_model_context_excluded


def client_event(chunk: ChatCompletionChunk) -> dict[str, object] | None:
    """Return a validated client event from a completion chunk."""
    extension = (chunk.model_extra or {}).get(LGOS_EXTENSION_KEY)
    if not isinstance(extension, dict):
        return None

    try:
        parsed = ClientEventExtension.model_validate(extension)
    except ValidationError:
        return None
    return parsed.event.model_dump(mode="json")


class ClientEventRenderer:
    """Render one live-updating custom element for a completion."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._element: cl.CustomElement | None = None

    async def render(self, chunk: ChatCompletionChunk) -> None:
        event = client_event(chunk)
        if event is None:
            return

        self._events.append(event)
        props = {"events": [*self._events]}
        if self._element is None:
            self._element = cl.CustomElement(
                name="ClientEventTimeline",
                props=props,
            )
            message = cl.Message(content="", elements=[self._element])
            mark_model_context_excluded(message)
            await message.send()
            return

        self._element.props = props
        await self._element.update()
