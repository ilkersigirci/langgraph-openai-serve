"""Public events emitted by LangGraph nodes and tools."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from langgraph_openai_serve.protocol import (
    CLIENT_EVENT_SCHEMA_VERSION,
    CLIENT_EVENT_TYPE,
)

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

    type: Literal["lgos.client_event"] = Field(
        description="Envelope type discriminator.",
    )
    schema_version: Literal[1] = Field(description="Client-event schema version.")
    event: _ClientEventData = Field(description="Public event exposed to clients.")


class StatusEventData(BaseModel):
    """Validated graph status used to render Responses commentary."""

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
        type=CLIENT_EVENT_TYPE,
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
    data = StatusEventData(
        description=description,
        done=done,
        hidden=hidden,
    )
    return client_event(
        "status",
        data.model_dump(mode="json"),
        namespace=namespace,
    )


def parse_status_event(value: object) -> StatusEventData | None:
    """Read a public graph status, ignoring private or diagnostic custom data."""
    if not isinstance(value, dict) or value.get("type") != CLIENT_EVENT_TYPE:
        return None

    try:
        envelope = _ClientEventEnvelope.model_validate(value)
        if envelope.event.type != "status":
            return None
        return StatusEventData.model_validate(envelope.event.data)
    except ValidationError:
        return None
