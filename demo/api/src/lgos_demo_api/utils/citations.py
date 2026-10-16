"""Add trusted URL citations to demo graph messages."""

import re
from collections.abc import Iterable, Mapping

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.content import Citation, create_citation

# Images are matched first so the URLs inside them are skipped, not cited.
_REFERENCE = re.compile(
    r"!\[[^]\r\n]*\]\([^)\r\n]+\)"
    r"|\[(?P<label>[^]\r\n]+)\]\((?P<url>[^)\r\n]+)\)"
    r"|(?P<bare>https?://[^\s<>()\[\]`]+)"
)


def _citations(text: str, sources: Mapping[str, str]) -> list[Citation]:
    citations = []
    for match in _REFERENCE.finditer(text):
        if match["url"] is not None:
            url, start, end = match["url"], match.start("label"), match.end("label")
        elif match["bare"] is not None:
            # A bare URL ending a sentence keeps its punctuation outside.
            url = match["bare"].rstrip(".,;:!?")
            start = match.start("bare")
            end = start + len(url)
        else:
            continue
        if url in sources:
            citations.append(
                create_citation(
                    url=url,
                    title=sources[url],
                    start_index=start,
                    end_index=end - 1,
                )
            )
    return citations


def tool_sources(messages: Iterable[BaseMessage], tool_name: str) -> dict[str, str]:
    """Collect the URL-to-title artifacts that a search tool's results carry."""
    sources: dict[str, str] = {}
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and message.name == tool_name
            and isinstance(message.artifact, dict)
        ):
            sources.update(message.artifact)
    return sources


def cite_sources(message: AIMessage, sources: Mapping[str, str]) -> AIMessage:
    """Annotate Markdown links and bare URLs that exactly match trusted sources."""
    blocks = []
    for block in message.content_blocks:
        if block["type"] != "text":
            blocks.append(block)
            continue
        citations = _citations(block["text"], sources)
        blocks.append(
            {**block, "annotations": [*block.get("annotations", []), *citations]}
        )
    # The copied content now contains LangChain-standard blocks, not provider data.
    return message.model_copy(
        update={
            "content": blocks,
            "response_metadata": {
                **message.response_metadata,
                "output_version": "v1",
            },
        }
    )


__all__ = ["cite_sources", "tool_sources"]
