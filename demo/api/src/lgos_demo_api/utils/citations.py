"""Add trusted URL citations to demo graph messages."""

import re
from collections.abc import Mapping

from langchain_core.messages import AIMessage
from langchain_core.messages.content import create_citation


def cite_markdown_links(
    message: AIMessage,
    sources: Mapping[str, str],
) -> AIMessage:
    """Annotate Markdown links whose URLs exactly match trusted sources."""
    blocks = []
    for block in message.content_blocks:
        if block["type"] != "text":
            blocks.append(block)
            continue
        citations = [
            create_citation(
                url=match.group("url"),
                title=sources[match.group("url")],
                start_index=match.start("label"),
                end_index=match.end("label") - 1,
            )
            for match in re.finditer(
                r"(?<!!)\[(?P<label>[^]\r\n]+)\]\((?P<url>[^)\r\n]+)\)",
                block["text"],
            )
            if match.group("url") in sources
        ]
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


__all__ = ["cite_markdown_links"]
