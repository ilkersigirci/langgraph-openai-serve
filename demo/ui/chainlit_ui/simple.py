"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, cast

import chainlit as cl
from demo.api.settings import settings
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_message import Annotation

client = AsyncOpenAI(
    base_url=settings.CHAINLIT_OPENAI_BASE_URL,
    api_key="DUMMY",
)


@dataclass(frozen=True)
class CitationSource:
    """A uniquely named Chainlit source element."""

    name: str
    title: str
    url: str

    @property
    def content(self) -> str:
        return f"[{self.title}]({self.url})"


def chunk_annotations(chunk: ChatCompletionChunk) -> list[Annotation]:
    """Read annotation deltas, including fields preserved as SDK extras."""
    if not chunk.choices:
        return []
    raw_annotations = getattr(chunk.choices[0].delta, "annotations", None) or []
    return [Annotation.model_validate(annotation) for annotation in raw_annotations]


def citation_sources(annotations: list[Annotation]) -> list[CitationSource]:
    """Return URL-deduplicated sources with unique element names."""
    sources: list[CitationSource] = []
    seen_urls: set[str] = set()
    for annotation in annotations:
        citation = annotation.url_citation
        if citation.url in seen_urls:
            continue
        seen_urls.add(citation.url)
        sources.append(
            CitationSource(
                name=f"Source {len(sources) + 1}",
                title=citation.title,
                url=citation.url,
            )
        )
    return sources


@cl.set_chat_profiles
async def set_chat_profiles() -> list[cl.ChatProfile]:
    models = await client.models.list()

    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description=f"Talk to `{model.id}` from the demo backend.",
        )
        for model in models.data
    ]


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="About",
            message="Tell me about yourself.",
            icon="",
        ),
        cl.Starter(
            label="History",
            message="Remember that my favorite color is green.",
            icon="",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("messages", [])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    model = cl.user_session.get("chat_profile")

    messages: list[dict[str, Any]] = [
        *(cl.user_session.get("messages") or []),
        {"role": "user", "content": message.content},
    ]
    cl.user_session.set("messages", messages)

    assistant_message = cl.Message(content="")
    annotations: list[Annotation] = []
    stream = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            annotations.extend(chunk_annotations(chunk))
            token = chunk.choices[0].delta.content or ""
            if token:
                await assistant_message.stream_token(token)

        messages.append({"role": "assistant", "content": assistant_message.content})
        cl.user_session.set("messages", messages)

        sources = citation_sources(annotations)
        if sources:
            assistant_message.content += "\n\nSources: " + ", ".join(
                source.name for source in sources
            )
            # Current Chainlit stubs do not accept concrete Text elements here.
            assistant_message.elements = cast(
                Any,
                [
                    cl.Text(
                        name=source.name,
                        content=source.content,
                        display="side",
                    )
                    for source in sources
                ],
            )

        await assistant_message.update()
    except asyncio.CancelledError:
        if assistant_message.content:
            await assistant_message.update()
        raise
    except Exception as exc:
        error = f"Chat completion failed: {exc}"
        if assistant_message.content:
            assistant_message.content = f"{assistant_message.content}\n\n{error}"
            await assistant_message.update()
        else:
            await cl.Message(content=error).send()
    finally:
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()
