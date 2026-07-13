import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import chainlit as cl
from demo.ui.chainlit_ui.simple import (
    attach_citations,
    chunk_annotations,
    citation_elements,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_message import Annotation

ANNOTATION_COUNT = 3


def test_chainlit_binds_annotation_spans_to_native_source_links() -> None:
    annotation = {
        "type": "url_citation",
        "url_citation": {
            "start_index": 0,
            "end_index": 3,
            "title": "Source",
            "url": "https://example.com/source",
        },
    }
    same_title_different_url = {
        **annotation,
        "url_citation": {
            **annotation["url_citation"],
            "start_index": 4,
            "end_index": 7,
            "url": "https://example.com/other",
        },
    }
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "citation-events",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "annotations": [
                            annotation,
                            annotation,
                            same_title_different_url,
                        ]
                    },
                }
            ],
        }
    )

    annotations = chunk_annotations(chunk)
    elements = citation_elements("[1] [2]", annotations, "thread-1")

    assert len(annotations) == ANNOTATION_COUNT
    assert [
        (element.name, element.content, element.display, element.thread_id)
        for element in elements
    ] == [
        (
            "[1]",
            "[Source](https://example.com/source)",
            "side",
            "thread-1",
        ),
        (
            "[1]",
            "[Source](https://example.com/source)",
            "side",
            "thread-1",
        ),
        (
            "[2]",
            "[Source](https://example.com/other)",
            "side",
            "thread-1",
        ),
    ]


def test_chainlit_closes_auto_opened_sidebar_after_attaching_citations(
    monkeypatch,
) -> None:
    annotation = {
        "type": "url_citation",
        "url_citation": {
            "start_index": 0,
            "end_index": 3,
            "title": "Source",
            "url": "https://example.com/source",
        },
    }
    update_message = AsyncMock()
    message = cast(
        cl.Message,
        SimpleNamespace(
            content="[1]",
            thread_id="thread-1",
            elements=[],
            update=update_message,
        ),
    )
    close_sidebar = AsyncMock()
    monkeypatch.setattr(
        "demo.ui.chainlit_ui.simple.cl.ElementSidebar.set_elements",
        close_sidebar,
    )

    asyncio.run(
        attach_citations(
            message,
            [Annotation.model_validate(annotation)],
        )
    )

    update_message.assert_awaited_once_with()
    close_sidebar.assert_awaited_once_with([])
