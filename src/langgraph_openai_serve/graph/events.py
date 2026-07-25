"""Public events emitted by LangGraph nodes and tools."""

from typing import Literal

from openai.types.chat.chat_completion_message import Annotation, AnnotationURLCitation
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

CLIENT_EVENT_SCHEMA_VERSION = 1
_CLIENT_EVENT_ENVELOPE_TYPE = "langgraph_openai_serve.client_event"

ClientEventType = Literal["status", "progress", "artifact"]


class _ClientEventData(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    type: ClientEventType = Field(description="Kind of client event.")
    namespace: tuple[str, ...] = Field(
        default=(),
        description="Author-defined path used to group related events.",
    )
    data: JsonValue = Field(description="JSON-safe event payload.")


class _ClientEventEnvelope(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    type: Literal["langgraph_openai_serve.client_event"] = Field(
        description="Envelope type discriminator.",
    )
    schema_version: Literal[1] = Field(description="Client-event schema version.")
    event: _ClientEventData = Field(description="Public event exposed to clients.")


class _StatusEventData(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    description: str = Field(
        min_length=1,
        description="User-facing status text.",
    )
    done: bool = Field(
        default=False,
        description="Whether the reported work is complete.",
    )
    hidden: bool = Field(
        default=False,
        description="Whether clients should hide the status.",
    )


def client_event(
    event_type: ClientEventType,
    data: JsonValue,
    *,
    namespace: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build an explicitly public, JSON-safe client stream event."""
    envelope = _ClientEventEnvelope(
        type=_CLIENT_EVENT_ENVELOPE_TYPE,
        schema_version=CLIENT_EVENT_SCHEMA_VERSION,
        event=_ClientEventData(
            type=event_type,
            namespace=namespace,
            data=data,
        ),
    )
    return envelope.model_dump(mode="json")


def status_event(
    description: str,
    *,
    done: bool = False,
    hidden: bool = False,
    namespace: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a portable status update for native client UI."""
    data = _StatusEventData(
        description=description,
        done=done,
        hidden=hidden,
    )
    return client_event(
        "status",
        data.model_dump(mode="json"),
        namespace=namespace,
    )


def client_event_extension(value: object) -> dict[str, object] | None:
    """Build a stream extension from validated public custom stream data."""
    if not isinstance(value, dict) or value.get("type") != _CLIENT_EVENT_ENVELOPE_TYPE:
        return None

    try:
        envelope = _ClientEventEnvelope.model_validate(value)
    except ValidationError:
        return None
    return envelope.model_dump(mode="json", exclude={"type"})


def citation_event(
    *,
    url: str,
    title: str,
    span: tuple[int, int],
) -> dict[str, object]:
    """Build an OpenAI URL citation from a Python-style half-open span."""
    start, stop = span
    if not 0 <= start < stop:
        raise ValueError("citation span must define a valid non-empty text range")

    return Annotation(
        type="url_citation",
        url_citation=AnnotationURLCitation(
            url=url,
            title=title,
            start_index=start,
            end_index=stop - 1,
        ),
    ).model_dump(mode="json")


def citation_slice(annotation: Annotation, content: str) -> slice:
    """Convert an OpenAI inclusive citation span to a validated Python slice."""
    citation = annotation.url_citation
    start = citation.start_index
    stop = citation.end_index + 1
    if not 0 <= start < stop <= len(content):
        raise ValueError("citation indices must refer to the final assistant text")
    return slice(start, stop)
