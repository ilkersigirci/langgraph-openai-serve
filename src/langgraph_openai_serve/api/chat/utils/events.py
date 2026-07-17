"""Adapt generic LangGraph custom events to OpenAI chat fields."""

from langgraph.types import CustomStreamPart
from openai.types.chat.chat_completion_message import Annotation

from langgraph_openai_serve.graph.events import (
    citation_slice,
    client_event_extension,
)

STREAM_EVENTS_METADATA_KEY = "langgraph_stream_events"
STREAM_EVENTS_METADATA_VALUE = "v1"


def stream_events_requested(metadata: dict[str, str] | None) -> bool:
    """Return whether a request opted into the supported event stream version."""
    return (metadata or {}).get(STREAM_EVENTS_METADATA_KEY) == (
        STREAM_EVENTS_METADATA_VALUE
    )


def client_event_extension_from_custom_event(
    event: CustomStreamPart,
) -> dict[str, object] | None:
    """Validate an explicitly public event and build its stream extension."""
    # Ignore LangGraph's execution namespace, which contains dynamic task IDs.
    # The public namespace is authored explicitly inside the validated event.
    return client_event_extension(event["data"])


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
