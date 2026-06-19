"""Transcript normalization for UI-event assistant messages."""

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage, Role
from langgraph_openai_serve.ui_events.codec import semantic_text_from_content


def normalize_ui_event_messages(
    messages: list[ChatCompletionRequestMessage],
) -> list[ChatCompletionRequestMessage]:
    """Replace prior UI-event assistant content with semantic text only."""
    normalized = []
    for message in messages:
        if message.role != Role.ASSISTANT:
            normalized.append(message)
            continue

        semantic_text = semantic_text_from_content(message.content)
        if semantic_text is None:
            normalized.append(message)
            continue

        normalized.append(message.model_copy(update={"content": semantic_text}))
    return normalized
