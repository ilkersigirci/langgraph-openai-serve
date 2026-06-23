"""Chat completion service."""

import logging
import time
from typing import AsyncIterator

from langgraph_openai_serve.api.chat.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
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

        completion = await invoke_run(run)
        tokens_used = usage_for(completion, chat_request.messages)

        response = chat_completion_response(
            model=chat_request.model,
            completion=completion,
            usage=tokens_used,
        )

        logger.info(
            f"Chat completion finished in {time.time() - start_time:.2f}s. "
            f"Total tokens: {tokens_used['total_tokens']}"
        )

        return response

    async def stream_completion(
        self, chat_request: ChatCompletionRequest, run: GraphRun
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        start_time = time.time()
        response_builder = ChatCompletionStreamResponseBuilder(chat_request.model)

        try:
            yield response_builder.role()

            async for event in stream_run(run):
                if isinstance(event, LangGraphInterrupt):
                    yield response_builder.interrupt(event)
                    yield response_builder.finish("tool_calls")
                    yield response_builder.done()
                    return

                yield response_builder.text(event)

            yield response_builder.finish("stop")
            yield response_builder.done()

            logger.info(
                f"Streamed chat completion finished in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logger.exception("Error streaming chat completion")
            yield response_builder.error(str(e))
            yield response_builder.done()
