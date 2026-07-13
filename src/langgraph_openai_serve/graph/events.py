"""OpenAI-compatible custom events emitted by LangGraph nodes and tools."""

from openai.types.chat.chat_completion_message import Annotation, AnnotationURLCitation


def citation_event(
    *,
    url: str,
    title: str,
    span: tuple[int, int],
) -> dict[str, object]:
    """Build an OpenAI URL citation from a Python-style half-open span."""
    start, stop = span
    if not 0 <= start < stop:
        raise ValueError("citation span must define a valid non-empty text range")

    return Annotation(
        type="url_citation",
        url_citation=AnnotationURLCitation(
            url=url,
            title=title,
            start_index=start,
            end_index=stop - 1,
        ),
    ).model_dump(mode="json")


def citation_slice(annotation: Annotation, content: str) -> slice:
    """Convert an OpenAI inclusive citation span to a validated Python slice."""
    citation = annotation.url_citation
    start = citation.start_index
    stop = citation.end_index + 1
    if not 0 <= start < stop <= len(content):
        raise ValueError("citation indices must refer to the final assistant text")
    return slice(start, stop)
