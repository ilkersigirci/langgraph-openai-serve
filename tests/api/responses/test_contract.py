"""Guard the supported SDK field matrix and golden streaming wire fixtures."""

import inspect
import json
from pathlib import Path
from typing import cast

import pytest
from openai.resources.responses.responses import AsyncResponses
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

FIXTURES = Path(__file__).with_name("fixtures")

SUPPORTED_REQUEST_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "metadata",
        "model",
        "parallel_tool_calls",
        "store",
        "stream",
        "text",
        "tool_choice",
        "tools",
        "user",
    }
)
UNSUPPORTED_REQUEST_FIELDS = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "max_output_tokens",
        "max_tool_calls",
        "moderation",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "stream_options",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
    }
)
SDK_TRANSPORT_FIELDS = frozenset(
    {"self", "extra_body", "extra_headers", "extra_query", "timeout"}
)


def _load_json(name: str) -> object:
    with FIXTURES.joinpath(name).open(encoding="utf-8") as fixture:
        return json.load(fixture)


def test_supported_request_matrix_covers_locked_sdk() -> None:
    sdk_fields = frozenset(inspect.signature(AsyncResponses.create).parameters)

    assert SUPPORTED_REQUEST_FIELDS.isdisjoint(UNSUPPORTED_REQUEST_FIELDS)
    assert sdk_fields == (
        SUPPORTED_REQUEST_FIELDS | UNSUPPORTED_REQUEST_FIELDS | SDK_TRANSPORT_FIELDS
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_event_types"),
    [
        pytest.param(
            "text_stream.json",
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
            id="commentary-and-final-text",
        ),
        pytest.param(
            "function_call_stream.json",
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
            id="function-call",
        ),
        pytest.param(
            "failed_stream.json",
            [
                "response.created",
                "response.in_progress",
                "error",
                "response.failed",
            ],
            id="failed",
        ),
    ],
)
def test_stream_fixtures_match_public_event_types(
    fixture_name: str,
    expected_event_types: list[str],
) -> None:
    payloads = cast("list[object]", _load_json(fixture_name))
    adapter = TypeAdapter(ResponseStreamEvent)

    events = [adapter.validate_python(payload) for payload in payloads]

    assert [event.type for event in events] == expected_event_types
    assert [event.sequence_number for event in events] == list(range(len(events)))
