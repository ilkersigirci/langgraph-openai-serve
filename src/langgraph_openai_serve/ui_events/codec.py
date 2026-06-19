"""Codec for AG-UI events carried through OpenAI-compatible content."""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterable

RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
CUSTOM = "CUSTOM"
STATE_SNAPSHOT = "STATE_SNAPSHOT"
STATE_DELTA = "STATE_DELTA"


class UIEventCodec:
    """Create and frame AG-UI event envelopes as complete NDJSON lines."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        message_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4()}"
        self.message_id = message_id or f"msg-{uuid.uuid4()}"
        self.thread_id = thread_id
        self._citation_map: dict[str, str] = {}

    def event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        """Build an AG-UI event."""
        event = {"type": event_type}
        event.update(
            {key: value for key, value in payload.items() if value is not None}
        )
        return event

    def run_start(self) -> dict[str, Any]:
        return self.event(
            RUN_STARTED,
            threadId=self.thread_id or "",
            runId=self.run_id,
        )

    def run_finish(self, *, finish_reason: str = "stop") -> dict[str, Any]:
        return self.event(
            RUN_FINISHED,
            threadId=self.thread_id or "",
            runId=self.run_id,
            result={"finishReason": finish_reason},
        )

    def run_error(self, message: str, *, code: str = "server_error") -> dict[str, Any]:
        return self.event(RUN_ERROR, message=message, code=code)

    def text_start(self) -> dict[str, Any]:
        return self.event(
            TEXT_MESSAGE_START,
            messageId=self.message_id,
            role="assistant",
        )

    def text_content(
        self,
        text: str,
        *,
        citations: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.event(
            TEXT_MESSAGE_CONTENT,
            messageId=self.message_id,
            delta=text,
            citations=citations,
        )

    def text_end(self) -> dict[str, Any]:
        return self.event(TEXT_MESSAGE_END, messageId=self.message_id)

    def custom(
        self,
        name: str,
        value: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata:
            if isinstance(value, dict):
                value = {**value, "metadata": metadata}
            else:
                value = {"value": value, "metadata": metadata}
        return self.event(CUSTOM, name=name, value=value)

    def state_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.event(STATE_SNAPSHOT, snapshot=state)

    def state_delta(
        self, delta: list[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, Any]:
        return self.event(
            STATE_DELTA, delta=delta if isinstance(delta, list) else [delta]
        )

    def citation(
        self,
        original_id: str,
        *,
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a normalized run-local citation definition event."""
        citation_id = self.normalize_citation_id(original_id)
        merged_metadata = {"originalId": original_id}
        if metadata:
            merged_metadata.update(metadata)
        if provenance is not None:
            merged_metadata["provenance"] = provenance
        return self.custom(
            "citation",
            {"id": citation_id},
            metadata=merged_metadata,
        )

    def normalize_citation_id(self, original_id: str) -> str:
        if original_id not in self._citation_map:
            self._citation_map[original_id] = f"cite-{len(self._citation_map) + 1}"
        return self._citation_map[original_id]

    def line(self, event: dict[str, Any]) -> str:
        return json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n"

    def lines(self, events: Iterable[dict[str, Any]]) -> str:
        return "".join(self.line(event) for event in events)

    def text_event_log(self, text: str, *, finish_reason: str = "stop") -> str:
        events = [
            self.run_start(),
            self.text_start(),
        ]
        if text:
            events.append(self.text_content(text))
        events.extend(
            [
                self.text_end(),
                self.run_finish(finish_reason=finish_reason),
            ]
        )
        return self.lines(events)


def parse_event_lines(content: str) -> list[dict[str, Any]]:
    """Parse UI-event NDJSON content into event dictionaries."""
    events = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError("UI-event line must be a JSON object")
        events.append(value)
    return events


def is_ag_ui_start(event: dict[str, Any]) -> bool:
    return event.get("type") == RUN_STARTED and isinstance(event.get("runId"), str)


def is_ui_event_content(content: str | None) -> bool:
    """Return true only when content starts with an AG-UI run event."""
    if not content:
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return isinstance(event, dict) and is_ag_ui_start(event)
    return False


def semantic_text_from_events(events: Iterable[dict[str, Any]]) -> str:
    """Concatenate only model-visible text-bearing events."""
    parts = []
    for event in events:
        if event.get("type") == TEXT_MESSAGE_CONTENT and isinstance(
            event.get("delta"), str
        ):
            parts.append(event["delta"])
    return "".join(parts)


def semantic_text_from_content(content: str | None) -> str | None:
    """Return semantic assistant text from UI-event content, or None when not UI-event."""
    if not is_ui_event_content(content):
        return None
    return semantic_text_from_events(parse_event_lines(content or ""))
