"""Adapt generic LangGraph custom events to OpenAI chat fields."""

from langgraph.types import CustomStreamPart
from openai.types.chat.chat_completion_message import Annotation

from langgraph_openai_serve.graph.events import citation_slice


def annotation_from_custom_event(
    event: CustomStreamPart,
    content: str,
) -> Annotation | None:
    """Validate a recognized OpenAI annotation custom event."""
    payload = event["data"]
    if not isinstance(payload, dict) or payload.get("type") != "url_citation":
        return None

    annotation = Annotation.model_validate(payload)
    citation_slice(annotation, content)
    return annotation
