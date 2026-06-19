import json

import pytest

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequestMessage,
    Role,
)
from langgraph_openai_serve.ui_events.codec import (
    UIEventCodec,
    is_ui_event_content,
    parse_event_lines,
    semantic_text_from_content,
)
from langgraph_openai_serve.ui_events.hitl import parse_ui_event_tool_response
from langgraph_openai_serve.ui_events.transcript import normalize_ui_event_messages


def test_text_event_log_has_markers_and_semantic_text() -> None:
    content = UIEventCodec(run_id="run-1", message_id="msg-1").text_event_log("hello")
    events = parse_event_lines(content)

    assert events[0]["type"] == "RUN_STARTED"
    assert events[1]["type"] == "TEXT_MESSAGE_START"
    assert events[-1]["type"] == "RUN_FINISHED"
    assert is_ui_event_content(content)
    assert semantic_text_from_content(content) == "hello"


def test_transcript_normalization_only_uses_marked_ui_event_content() -> None:
    event_content = UIEventCodec().text_event_log("semantic")
    ordinary_json = '{"type":"TEXT_MESSAGE_CONTENT","delta":"not a protocol log"}'
    messages = [
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content=event_content),
        ChatCompletionRequestMessage(role=Role.ASSISTANT, content=ordinary_json),
    ]

    normalized = normalize_ui_event_messages(messages)

    assert normalized[0].content == "semantic"
    assert normalized[1].content == ordinary_json


def test_citation_ids_are_run_local_and_preserve_provenance() -> None:
    codec = UIEventCodec()

    first = codec.citation("source-a", provenance={"url": "https://example.test/a"})
    second = codec.citation("source-a")
    third = codec.citation("source-b")

    assert first["value"]["id"] == "cite-1"
    assert second["value"]["id"] == "cite-1"
    assert third["value"]["id"] == "cite-2"
    assert first["value"]["metadata"]["originalId"] == "source-a"
    assert first["value"]["metadata"]["provenance"]["url"] == "https://example.test/a"


def test_hitl_tool_response_validation() -> None:
    message = ChatCompletionRequestMessage(
        role=Role.TOOL,
        tool_call_id="call-1",
        content=json.dumps({"runId": "run-1", "approved": True}),
    )

    assert parse_ui_event_tool_response(
        message,
        expected_tool_call_id="call-1",
        expected_run_id="run-1",
    )["approved"]

    with pytest.raises(ValueError):
        parse_ui_event_tool_response(message, expected_tool_call_id="call-2")
