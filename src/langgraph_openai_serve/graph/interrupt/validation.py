"""Validation shared by interrupt codecs and graph execution."""

import json
from typing import Any

from langgraph_openai_serve.graph.interrupt.errors import InvalidInterruptPayloadError


def validate_interrupt_payload(payload: Any) -> None:
    """Reject graph values that cannot cross a JSON protocol boundary."""
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = "LangGraph interrupt payloads must be valid JSON values."
        raise InvalidInterruptPayloadError(msg) from exc


__all__ = ["validate_interrupt_payload"]
