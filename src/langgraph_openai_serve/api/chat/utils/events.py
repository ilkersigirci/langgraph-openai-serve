"""Adapt generic LangGraph custom events to OpenAI chat fields."""

from langgraph.types import CustomStreamPart
from openai.types.chat.chat_completion_message import Annotation


def annotation_from_custom_event(
    event: CustomStreamPart,
    content: str,
) -> Annotation | None:
    """Validate a recognized OpenAI annotation custom event."""
    payload = event["data"]
    if not isinstance(payload, dict) or payload.get("type") != "url_citation":
        return None

    annotation = Annotation.model_validate(payload)
    citation = annotation.url_citation
    if (
        citation.start_index < 0
        or citation.end_index < citation.start_index
        or citation.end_index > len(content)
    ):
        raise ValueError("citation indices must refer to the final assistant text")

    return annotation
