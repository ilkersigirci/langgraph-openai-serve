from demo.ui.chainlit_ui.simple import (
    CitationSource,
    chunk_annotations,
    citation_sources,
)
from openai.types.chat import ChatCompletionChunk

ANNOTATION_COUNT = 3


def test_chainlit_converts_annotation_deltas_to_unique_sources() -> None:
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
    sources = citation_sources(annotations)

    assert len(annotations) == ANNOTATION_COUNT
    assert sources == [
        CitationSource(
            name="Source 1",
            title="Source",
            url="https://example.com/source",
        ),
        CitationSource(
            name="Source 2",
            title="Source",
            url="https://example.com/other",
        ),
    ]
    assert sources[0].content == "[Source](https://example.com/source)"
