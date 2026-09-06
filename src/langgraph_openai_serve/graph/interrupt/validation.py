"""Validate graph-authored LangGraph interrupt payloads."""

import json
from typing import Any

from langgraph_openai_serve.graph.interrupt.errors import InvalidInterruptPayloadError


def validate_interrupt_payload(payload: Any) -> None:
    """Require function-call arguments containing valid JSON object values."""
    if not isinstance(payload, dict):
        msg = "LangGraph interrupt payloads must be JSON objects."
        raise InvalidInterruptPayloadError(msg)
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = "LangGraph interrupt payloads must be valid JSON values."
        raise InvalidInterruptPayloadError(msg) from exc


__all__ = ["validate_interrupt_payload"]
