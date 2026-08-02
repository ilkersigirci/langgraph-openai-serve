from enum import StrEnum


class GraphFeature(StrEnum):
    """Features supported by a registered graph."""

    CLIENT_EVENTS = "client_events"
    INTERRUPTS = "interrupts"
