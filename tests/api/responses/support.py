import json
from pathlib import Path
from typing import Any, cast

from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

_FIXTURE_DIR = Path(__file__).with_name("fixtures")
_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)
_ID_PREFIXES = {
    "resp_": "response",
    "msg_": "message",
    "fc_": "function_call",
}


def normalize_stream_payloads(body: str) -> list[dict[str, Any]]:
    payloads = _parse_sse(body)
    for payload in payloads:
        _STREAM_EVENT_ADAPTER.validate_python(payload)
    return cast("list[dict[str, Any]]", _normalize(payloads))


def load_stream_fixture(fixture_name: str) -> list[dict[str, Any]]:
    return cast(
        "list[dict[str, Any]]",
        json.loads((_FIXTURE_DIR / fixture_name).read_text(encoding="utf-8")),
    )


def _parse_sse(body: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for frame in filter(None, body.split("\n\n")):
        event_line, data_line = frame.splitlines()
        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["type"] == event_line.removeprefix("event: ")
        payloads.append(payload)
    return payloads


def _normalize(value: Any, identifiers: dict[str, str] | None = None) -> Any:
    if identifiers is None:
        identifiers = {}
    if isinstance(value, dict):
        return {
            key: (
                _placeholder(f"{key}:{item!r}", key, identifiers)
                if key in {"created_at", "completed_at"} and item is not None
                else _normalize(item, identifiers)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize(item, identifiers) for item in value]
    if isinstance(value, str):
        for prefix, label in _ID_PREFIXES.items():
            if value.startswith(prefix):
                return _placeholder(value, label, identifiers)
    return value


def _placeholder(identity: str, label: str, identifiers: dict[str, str]) -> str:
    if identity not in identifiers:
        index = sum(
            placeholder.startswith(f"<{label}_") for placeholder in identifiers.values()
        )
        identifiers[identity] = f"<{label}_{index}>"
    return identifiers[identity]
