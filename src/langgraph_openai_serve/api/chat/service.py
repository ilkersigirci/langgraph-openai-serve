"""Chat completion service.

This module provides a service for handling chat completions, implementing
business logic that was previously in the router.
"""

import json
import logging
import time
import uuid
from typing import AsyncIterator

from langgraph_openai_serve.api.chat.schemas import (
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
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.runner import (
    run_langgraph,
    run_langgraph_stream,
)
from langgraph_openai_serve.hitl.openai import (
    HitlToolCall,
    tool_call_delta_payload,
    tool_call_message_payload,
)

logger = logging.getLogger(__name__)


class ChatCompletionService:
    """Service for handling chat completions."""

    async def generate_completion(
        self, chat_request: ChatCompletionRequest, graph_registry: GraphRegistry
    ) -> ChatCompletionResponse:
        """Generate a chat completion.

        Args:
            chat_request: The chat completion request.
            graph_registry: The GraphRegistry object containing registered graphs.

        Returns:
            A chat completion response.

        Raises:
            Exception: If there is an error generating the completion.
        """
        start_time = time.time()

        # Get the completion from the LangGraph model
        result = await run_langgraph(
            model=chat_request.model,
            messages=chat_request.messages,
            graph_registry=graph_registry,
            request=chat_request,
        )

        if result.tool_call is not None:
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=chat_request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatCompletionResponseMessage(
                            role=Role.ASSISTANT,
                            content=None,
                            tool_calls=[tool_call_message_payload(result.tool_call)],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(**result.token_usage),
            )

        completion = result.content
        tokens_used = result.token_usage

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
        self, chat_request: ChatCompletionRequest, graph_registry: GraphRegistry
    ) -> AsyncIterator[str]:
        """Stream a chat completion response.

        Args:
            chat_request: The chat completion request.
            graph_registry: The GraphRegistry object containing registered graphs.

        Yields:
            Chunks of the chat completion response.
        """
        start_time = time.time()
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        try:
            # Send the initial response with the role
            yield self._stream_chunk(
                response_id,
                created,
                chat_request.model,
                ChatCompletionStreamResponseDelta(role=Role.ASSISTANT),
            )

            finish_reason = "stop"

            # Stream text until the graph completes or pauses for human input.
            async for event in run_langgraph_stream(
                model=chat_request.model,
                messages=chat_request.messages,
                graph_registry=graph_registry,
                request=chat_request,
            ):
                if isinstance(event, HitlToolCall):
                    finish_reason = "tool_calls"
                    yield self._stream_chunk(
                        response_id,
                        created,
                        chat_request.model,
                        ChatCompletionStreamResponseDelta(
                            tool_calls=[tool_call_delta_payload(event)]
                        ),
                    )
                    break

                if not event:
                    continue
                yield self._stream_chunk(
                    response_id,
                    created,
                    chat_request.model,
                    ChatCompletionStreamResponseDelta(content=event),
                )

            # Send the final response with finish_reason
            yield self._stream_chunk(
                response_id,
                created,
                chat_request.model,
                ChatCompletionStreamResponseDelta(),
                finish_reason,
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

    def _stream_chunk(
        self,
        response_id: str,
        created: int,
        model: str,
        delta: ChatCompletionStreamResponseDelta,
        finish_reason: str | None = None,
    ) -> str:
        response = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )
        return self._format_stream_chunk(response)
