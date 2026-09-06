from typing import Any

import pytest

from langgraph_openai_serve.graph.events import (
    client_event,
    client_event_extension,
    status_event,
    status_event_data,
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
        "type": "langgraph_openai_serve.client_event",
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


def test_client_event_extension_and_status_event_data() -> None:
    event = status_event("Processing", namespace=("test",))
    extension = client_event_extension(event)
    assert extension is not None
    assert extension["schema_version"] == 1
    assert "event" in extension

    status_data = status_event_data(extension)
    assert status_data is not None
    assert status_data["description"] == "Processing"
    assert status_data["done"] is False
    assert status_data["hidden"] is False

    assert client_event_extension({"type": "wrong"}) is None
    assert status_event_data({"event": {"type": "other"}}) is None
