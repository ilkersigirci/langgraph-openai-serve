"""Functions for generating chat completions."""

from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing

from langchain_core.messages import AIMessage

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.utils.events import (
    client_event_extension_from_custom_event,
    stream_events_requested,
)
from langgraph_openai_serve.api.chat.utils.responses import (
    ChatCompletionStreamResponseBuilder,
    annotations_from_message,
    chat_completion_response,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.runner import (
    invoke_run,
    stream_run,
)
from langgraph_openai_serve.graph.utils import GraphRun

logger = get_logger(__name__)


async def generate_completion(
    chat_request: ChatCompletionRequest, run: GraphRun
) -> ChatCompletionResponse:
    """Generate a chat completion."""
    invocation = await invoke_run(run)
    return chat_completion_response(
        model=chat_request.model,
        completion=invocation.output,
    )


async def stream_completion(
    chat_request: ChatCompletionRequest, run: GraphRun
) -> AsyncGenerator[str, None]:
    """
    Stream a chat completion response.

    Yields:
        String chunks representing Server-Sent Events.

    """
    include_usage = bool(
        chat_request.stream_options is not None
        and chat_request.stream_options.include_usage
    )
    response_builder = ChatCompletionStreamResponseBuilder(
        chat_request.model,
        include_usage=include_usage,
    )
    final_message: AIMessage | None = None
    text_parts: list[str] = []
    include_client_events = run.config.supports(
        GraphFeature.CLIENT_EVENTS
    ) and stream_events_requested(chat_request.metadata)

    try:  # ruff: ignore[too-many-nested-blocks, too-many-statements-in-try-clause]
        yield response_builder.role()

        run_stream = stream_run(run)
        # Closing the HTTP response must also close the nested graph stream.
        async with aclosing(run_stream):
            async for event in run_stream:
                if isinstance(event, LangGraphInterruptBatch):
                    yield response_builder.interrupt(event)
                    yield response_builder.finish("tool_calls")
                    yield response_builder.done()
                    return

                if isinstance(event, AIMessage):
                    final_message = event
                    continue

                if not isinstance(event, str):
                    if include_client_events:
                        extension = client_event_extension_from_custom_event(event)
                        if extension is not None:
                            yield response_builder.client_event(extension)
                    continue

                text_parts.append(event)
                yield response_builder.text(event)

        final_message = _require_final_message(final_message)
        for chunk in _final_chunks(
            response_builder,
            final_message,
            streamed_text="".join(text_parts) if text_parts else None,
            include_usage=include_usage,
        ):
            yield chunk

    except Exception:
        logger.exception("chat_completion.stream_failed")
        yield response_builder.error("Internal server error")
        yield response_builder.done()


def _require_final_message(message: AIMessage | None) -> AIMessage:
    if message is None:
        msg = "LangGraph stream completed without a final assistant message."
        raise RuntimeError(msg)
    return message


def _final_chunks(
    response_builder: ChatCompletionStreamResponseBuilder,
    message: AIMessage,
    *,
    streamed_text: str | None,
    include_usage: bool,
) -> Iterator[str]:
    message_text = str(message.text)
    if streamed_text is None:
        if message_text:
            yield response_builder.text(message_text)
    elif streamed_text != message_text:
        msg = "Streamed assistant text did not match the final assistant message."
        raise RuntimeError(msg)
    finish_reason = "tool_calls" if message.tool_calls else "stop"
    if message.tool_calls:
        yield response_builder.tool_calls(message)
    yield response_builder.finish(
        finish_reason,
        annotations=annotations_from_message(message),
    )
    if include_usage and message.usage_metadata is not None:
        yield response_builder.usage(message.usage_metadata)
    yield response_builder.done()
