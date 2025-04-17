"""Chat completion service.

This module provides a service for handling chat completions, implementing
business logic that was previously in the router.
"""

import json
import logging
import time
import uuid
from typing import AsyncIterator

from langgraph_openai_serve.graph.runner import (
    run_langgraph,
    run_langgraph_stream,
)
from langgraph_openai_serve.schemas.openai_schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    Role,
    UsageInfo,
)

logger = logging.getLogger(__name__)


class ChatCompletionService:
    """Service for handling chat completions."""

    async def generate_completion(
        self, chat_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion.

        Args:
            chat_request: The chat completion request.

        Returns:
            A chat completion response.

        Raises:
            Exception: If there is an error generating the completion.
        """
        start_time = time.time()

        # Get the completion from the LangGraph model
        completion, tokens_used = await run_langgraph(
            model=chat_request.model,
            messages=chat_request.messages,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            tools=chat_request.tools,
        )

        # Build the response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=chat_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        role=Role.ASSISTANT,
                        content=completion,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=tokens_used["prompt_tokens"],
                completion_tokens=tokens_used["completion_tokens"],
                total_tokens=tokens_used["total_tokens"],
            ),
        )

        logger.info(
            f"Chat completion finished in {time.time() - start_time:.2f}s. "
            f"Total tokens: {tokens_used['total_tokens']}"
        )

        return response

    async def stream_completion(
        self, chat_request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Stream a chat completion response.

        Args:
            chat_request: The chat completion request.

        Yields:
            Chunks of the chat completion response.
        """
        start_time = time.time()
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        try:
            # Send the initial response with the role
            yield self._format_stream_chunk(
                ChatCompletionStreamResponse(
                    id=response_id,
                    created=created,
                    model=chat_request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=ChatCompletionStreamResponseDelta(
                                role=Role.ASSISTANT,
                            ),
                            finish_reason=None,
                        )
                    ],
                )
            )

            # Stream the completion from the LangGraph model
            async for chunk, _ in run_langgraph_stream(
                model=chat_request.model,
                messages=chat_request.messages,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                tools=chat_request.tools,
            ):
                # Send the content chunk
                yield self._format_stream_chunk(
                    ChatCompletionStreamResponse(
                        id=response_id,
                        created=created,
                        model=chat_request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=ChatCompletionStreamResponseDelta(
                                    content=chunk,
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                )

            # Send the final response with finish_reason
            yield self._format_stream_chunk(
                ChatCompletionStreamResponse(
                    id=response_id,
                    created=created,
                    model=chat_request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=ChatCompletionStreamResponseDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
            )

            # Send the [DONE] message
            yield "data: [DONE]\n\n"

            logger.info(
                f"Streamed chat completion finished in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logger.exception("Error streaming chat completion")
            # In case of an error, send an error message
            error_response = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"

    def _format_stream_chunk(self, response: ChatCompletionStreamResponse) -> str:
        """Format a stream chunk as a server-sent event.

        Args:
            response: The response to format.

        Returns:
            The formatted server-sent event.
        """
        return f"data: {json.dumps(response.model_dump())}\n\n"
