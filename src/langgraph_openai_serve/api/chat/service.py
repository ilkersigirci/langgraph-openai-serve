"""Chat completion service."""

import logging
import time
from contextlib import aclosing
from typing import AsyncGenerator

from langgraph.types import CustomStreamPart

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from langgraph_openai_serve.api.chat.utils.events import annotation_from_custom_event
from langgraph_openai_serve.api.chat.utils.responses import (
    ChatCompletionStreamResponseBuilder,
    chat_completion_response,
)
from langgraph_openai_serve.graph.runner import (
    LangGraphInterrupt,
    invoke_run,
    stream_run,
    usage_for,
)
from langgraph_openai_serve.graph.utils import GraphRun

logger = logging.getLogger(__name__)


class ChatCompletionService:
    """Service for handling chat completions."""

    async def generate_completion(
        self, chat_request: ChatCompletionRequest, run: GraphRun
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
        self, chat_request: ChatCompletionRequest, run: GraphRun
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response."""
        start_time = time.time()
        response_builder = ChatCompletionStreamResponseBuilder(chat_request.model)
        custom_events: list[CustomStreamPart] = []
        content_parts: list[str] = []

        try:
            yield response_builder.role()

            run_stream = stream_run(run)
            # Closing the HTTP response must also close the nested graph stream.
            async with aclosing(run_stream):
                async for event in run_stream:
                    if isinstance(event, LangGraphInterrupt):
                        yield response_builder.interrupt(event)
                        yield response_builder.finish("tool_calls")
                        yield response_builder.done()
                        return

                    if not isinstance(event, str):
                        custom_events.append(event)
                        continue

                    content_parts.append(event)
                    yield response_builder.text(event)

            content = "".join(content_parts)
            annotations = [
                annotation
                for event in custom_events
                if (annotation := annotation_from_custom_event(event, content))
                is not None
            ]
            yield response_builder.finish("stop", annotations=annotations)
            yield response_builder.done()

            logger.info(
                f"Streamed chat completion finished in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logger.exception("Error streaming chat completion")
            yield response_builder.error(str(e))
            yield response_builder.done()
