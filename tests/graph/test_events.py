from typing import Any

import pytest

from langgraph_openai_serve.graph.events import (
    client_event,
    parse_status_event,
    status_event,
)


@pytest.mark.parametrize(
    ("event_type", "data", "namespace"),
    [
        pytest.param("debug", {}, (), id="unsupported-type"),
        pytest.param("status", {"value": object()}, (), id="non-json-data"),
        pytest.param("status", {}, (1,), id="non-string-namespace"),
    ],
)
def test_client_event_rejects_invalid_public_values(
    event_type: Any,
    data: Any,
    namespace: tuple[Any, ...],
) -> None:
    with pytest.raises(ValueError, match="validation error"):
        client_event(event_type, data, namespace=namespace)


def test_status_event_builds_the_portable_status_shape() -> None:
    assert status_event(
        "Generating audio",
        done=True,
        hidden=True,
        namespace=("media",),
    ) == {
        "type": "lgos.client_event",
        "schema_version": 1,
        "event": {
            "type": "status",
            "namespace": ["media"],
            "data": {
                "description": "Generating audio",
                "done": True,
                "hidden": True,
            },
        },
    }

    with pytest.raises(ValueError, match="validation error"):
        status_event("")


def test_parse_status_event_preserves_status_fields() -> None:
    event = status_event("Processing", namespace=("test",))
    status_data = parse_status_event(event)
    assert status_data is not None
    assert status_data.description == "Processing"
    assert status_data.done is False
    assert status_data.hidden is False

    assert parse_status_event({"type": "wrong"}) is None
    assert parse_status_event(client_event("progress", {"value": 1})) is None
    assert parse_status_event(client_event("status", {"description": ""})) is None
