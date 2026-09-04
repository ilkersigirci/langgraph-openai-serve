from enum import StrEnum


class GraphFeature(StrEnum):
    """Features supported by a registered graph."""

    CLIENT_EVENTS = "client_events"
    FILE_INPUTS = "file_inputs"
    INTERRUPTS = "interrupts"
