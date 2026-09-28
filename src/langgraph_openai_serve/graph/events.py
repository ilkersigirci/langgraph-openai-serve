"""Status updates that graph nodes publish as Responses commentary."""

from langgraph_openai_serve.protocol import STATUS_EVENT_TYPE


def status_event(description: str) -> dict[str, str]:
    """Build a status update for a node to write with ``get_stream_writer()``."""
    return {"type": STATUS_EVENT_TYPE, "description": description}


def status_description(value: object) -> str | None:
    """Return a status update's description, ignoring other custom stream data."""
    if not isinstance(value, dict) or value.get("type") != STATUS_EVENT_TYPE:
        return None
    description = value.get("description")
    return description if isinstance(description, str) and description else None
