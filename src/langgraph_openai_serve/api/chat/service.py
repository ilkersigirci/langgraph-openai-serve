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
from langgraph_openai_serve.ui_events.codec import UIEventCodec
from langgraph_openai_serve.ui_events.hitl import (
    UIEventInterrupt,
    make_ui_event_tool_call,
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
        ui_event_options = chat_request.ui_event_options()

        # Get the completion from the LangGraph model
        try:
            completion, tokens_used = await run_langgraph(
                model=chat_request.model,
                messages=chat_request.messages,
                graph_registry=graph_registry,
                request=chat_request,
            )
        except UIEventInterrupt as interrupt:
            if not ui_event_options.enabled:
                raise
            codec = UIEventCodec(thread_id=ui_event_options.thread_id)
            return self._build_hitl_tool_call_response(
                chat_request,
                codec,
                interrupt.interrupt,
            )
        content = completion
        if ui_event_options.enabled:
            content = UIEventCodec(
                thread_id=ui_event_options.thread_id,
            ).text_event_log(completion)

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
                        content=content,
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

    def _build_hitl_tool_call_response(
        self,
        chat_request: ChatCompletionRequest,
        codec: UIEventCodec,
        interrupt: object,
    ) -> ChatCompletionResponse:
        interrupt_value = getattr(interrupt, "value", interrupt)
        interrupt_id = getattr(interrupt, "id", None)
        event = codec.custom(
            name="hitl.request",
            value={
                "request": interrupt_value,
                "responseSchema": {
                    "type": "object",
                    "description": "UI-collected response to resume the interrupted graph.",
                },
            },
            metadata={"interruptId": interrupt_id} if interrupt_id else None,
        )
        tool_call = make_ui_event_tool_call(event)
        prompt_tokens = sum(
            len((m.content or "").split()) for m in chat_request.messages
        )
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
                        tool_calls=[tool_call],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
            ),
        )

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
        ui_event_options = chat_request.ui_event_options()

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

            if ui_event_options.enabled:
                async for event_chunk in self._stream_event_completion(
                    chat_request,
                    graph_registry,
                    response_id=response_id,
                    created=created,
                    thread_id=ui_event_options.thread_id,
                ):
                    yield event_chunk
                logger.info(
                    f"Streamed UI-event chat completion finished in "
                    f"{time.time() - start_time:.2f}s"
                )
                return

            # Stream the completion from the LangGraph model
            async for chunk, _ in run_langgraph_stream(
                model=chat_request.model,
                messages=chat_request.messages,
                graph_registry=graph_registry,
                request=chat_request,
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

    async def _stream_event_completion(
        self,
        chat_request: ChatCompletionRequest,
        graph_registry: GraphRegistry,
        *,
        response_id: str,
        created: int,
        thread_id: str | None,
    ) -> AsyncIterator[str]:
        codec = UIEventCodec(thread_id=thread_id)
        ag_ui_started = False
        text_started = False
        try:
            for event in [
                codec.run_start(),
                codec.text_start(),
            ]:
                ag_ui_started = True
                if event["type"] == "TEXT_MESSAGE_START":
                    text_started = True
                yield self._format_event_stream_chunk(
                    response_id,
                    created,
                    chat_request.model,
                    codec.line(event),
                )

            async for chunk, _ in run_langgraph_stream(
                model=chat_request.model,
                messages=chat_request.messages,
                graph_registry=graph_registry,
                request=chat_request,
            ):
                yield self._format_event_stream_chunk(
                    response_id,
                    created,
                    chat_request.model,
                    codec.line(codec.text_content(chunk)),
                )

            for event in [
                codec.text_end(),
                codec.run_finish(finish_reason="stop"),
            ]:
                yield self._format_event_stream_chunk(
                    response_id,
                    created,
                    chat_request.model,
                    codec.line(event),
                )

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
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Error streaming UI-event chat completion")
            if not ag_ui_started:
                error_response = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
                return

            error_events = []
            if text_started:
                error_events.append(codec.text_end())
            error_events.append(codec.run_error(str(e)))
            for event in error_events:
                yield self._format_event_stream_chunk(
                    response_id,
                    created,
                    chat_request.model,
                    codec.line(event),
                )
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
            yield "data: [DONE]\n\n"

    def _format_stream_chunk(self, response: ChatCompletionStreamResponse) -> str:
        """Format a stream chunk as a server-sent event.

        Args:
            response: The response to format.

        Returns:
            The formatted server-sent event.
        """
        return f"data: {json.dumps(response.model_dump())}\n\n"

    def _format_event_stream_chunk(
        self,
        response_id: str,
        created: int,
        model: str,
        content: str,
    ) -> str:
        return self._format_stream_chunk(
            ChatCompletionStreamResponse(
                id=response_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=ChatCompletionStreamResponseDelta(content=content),
                        finish_reason=None,
                    )
                ],
            )
        )
