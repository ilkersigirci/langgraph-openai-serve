"""Validate native LangChain citations independently of the HTTP protocol."""

from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.messages.content import Citation


def citation_slice(start_index: int, end_index: int, content: str) -> slice:
    """Convert an inclusive citation span to a validated Python slice."""
    stop = end_index + 1
    if not 0 <= start_index < stop <= len(content):
        msg = "citation indices must refer to the final assistant text"
        raise ValueError(msg)
    return slice(start_index, stop)


def citations_from_message(message: AIMessage) -> list[Citation]:
    """Extract URL citations with validated offsets into the complete text."""
    text = str(message.text)
    citations: list[Citation] = []
    text_offset = 0
    for block in message.content_blocks:
        if block["type"] != "text":
            continue
        for raw_citation in block.get("annotations", []):
            if raw_citation.get("type") != "citation":
                continue
            required = {"url", "title", "start_index", "end_index"}
            if not required.issubset(raw_citation):
                continue
            citation = cast("Citation", raw_citation).copy()
            citation["start_index"] += text_offset
            citation["end_index"] += text_offset
            span = citation_slice(citation["start_index"], citation["end_index"], text)
            cited_text = citation.get("cited_text")
            if cited_text is not None and text[span] != cited_text:
                msg = "citation indices must match cited_text"
                raise ValueError(msg)
            citations.append(citation)
        text_offset += len(block["text"])
    return citations
