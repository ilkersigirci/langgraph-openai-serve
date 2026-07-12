"""OpenAI-compatible custom events emitted by LangGraph nodes and tools."""

from openai.types.chat.chat_completion_message import Annotation, AnnotationURLCitation


def citation_event(
    *,
    url: str,
    title: str,
    start_index: int,
    end_index: int,
) -> dict[str, object]:
    """Build an OpenAI URL citation annotation for LangGraph's stream writer."""
    if start_index < 0 or end_index < start_index:
        raise ValueError("citation indices must define a valid text span")

    return Annotation(
        type="url_citation",
        url_citation=AnnotationURLCitation(
            url=url,
            title=title,
            start_index=start_index,
            end_index=end_index,
        ),
    ).model_dump(mode="json")
