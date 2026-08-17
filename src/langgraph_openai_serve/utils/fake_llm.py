"""Shared fake streaming model helpers for demos and tests."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage


async def stream_fake_chat_response(
    response: str,
    prompt: str,
) -> str:
    """Stream a deterministic fake chat response and collect it for graph state."""
    model = GenericFakeChatModel(messages=iter([response]))
    return "".join(
        [
            str(chunk.content)
            async for chunk in model.astream([HumanMessage(content=prompt)])
        ]
    )
