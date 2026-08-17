"""Functions for generating chat completions."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.utils.events import (
    annotation_from_custom_event,
    client_event_extension_from_custom_event,
    stream_events_requested,
)
from langgraph_openai_serve.api.chat.utils.responses import (
    ChatCompletionStreamResponseBuilder,
    chat_completion_response,
)
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.runner import (
    invoke_run,
    stream_run,
    usage_for,
)
from langgraph_openai_serve.graph.utils import GraphRun

if TYPE_CHECKING:
    from langgraph.types import CustomStreamPart

logger = logging.getLogger(__name__)


async def generate_completion(
    chat_request: ChatCompletionRequest, run: GraphRun
) -> ChatCompletionResponse:
    """Generate a chat completion."""
    start_time = time.time()

    invocation = await invoke_run(run)
    completion = invocation.output
    tokens_used = usage_for(completion, chat_request.messages)
    annotations = (
        [
            annotation
            for event in invocation.custom_events
            if (annotation := annotation_from_custom_event(event, completion))
            is not None
        ]
        if isinstance(completion, str)
        else []
    )

    response = chat_completion_response(
        model=chat_request.model,
        completion=completion,
        annotations=annotations,
        usage=tokens_used,
    )

    logger.info(
        f"Chat completion finished in {time.time() - start_time:.2f}s. "
        f"Total tokens: {tokens_used['total_tokens']}"
    )

    return response


async def stream_completion(
    chat_request: ChatCompletionRequest, run: GraphRun
) -> AsyncGenerator[str, None]:
    """Stream a chat completion response."""
    start_time = time.time()
    response_builder = ChatCompletionStreamResponseBuilder(chat_request.model)
    custom_events: list[CustomStreamPart] = []
    content_parts: list[str] = []
    include_client_events = run.config.supports(
        GraphFeature.CLIENT_EVENTS
    ) and stream_events_requested(chat_request.metadata)

    try:
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

                if not isinstance(event, str):
                    custom_events.append(event)
                    if include_client_events:
                        extension = client_event_extension_from_custom_event(event)
                        if extension is not None:
                            yield response_builder.client_event(extension)
                    continue

                content_parts.append(event)
                yield response_builder.text(event)

        content = "".join(content_parts)
        annotations = [
            annotation
            for event in custom_events
            if (annotation := annotation_from_custom_event(event, content)) is not None
        ]
        yield response_builder.finish("stop", annotations=annotations)
        yield response_builder.done()

        logger.info(
            f"Streamed chat completion finished in {time.time() - start_time:.2f}s"
        )

    except Exception:
        logger.exception("Error streaming chat completion")
        yield response_builder.error("Internal server error")
        yield response_builder.done()
