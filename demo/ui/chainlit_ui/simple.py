"""Simple Chainlit UI for the demo OpenAI-compatible LangGraph server."""

import asyncio
import contextlib
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


def chunk_annotations(chunk: ChatCompletionChunk) -> list[Annotation]:
    """Read annotation deltas, including fields preserved as SDK extras."""
    if not chunk.choices:
        return []
    raw_annotations = getattr(chunk.choices[0].delta, "annotations", None) or []
    return [Annotation.model_validate(annotation) for annotation in raw_annotations]


def citation_elements(
    answer: str,
    annotations: list[Annotation],
    thread_id: str,
) -> list[cl.Text]:
    """Bind annotated answer spans to native Chainlit source references."""
    elements: list[cl.Text] = []
    for annotation in annotations:
        citation = annotation.url_citation
        elements.append(
            cl.Text(
                thread_id=thread_id,
                name=answer[citation.start_index : citation.end_index],
                content=f"[{citation.title}]({citation.url})",
                display="side",
            )
        )
    return elements


async def attach_citations(
    message: cl.Message,
    annotations: list[Annotation],
) -> None:
    """Attach clickable sources without leaving Chainlit's sidebar open."""
    elements = citation_elements(message.content, annotations, message.thread_id)
    message.elements = cast(Any, elements)
    await message.update()

    if elements:
        await cl.ElementSidebar.set_elements([])


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

        await attach_citations(assistant_message, annotations)
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
