"""Prepare and execute graph runs for OpenAI Chat Completions."""

from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing

from langchain_core.messages import AIMessage
from openai.types.chat import ChatCompletion

from langgraph_openai_serve.api.chat.request import decode_chat_request
from langgraph_openai_serve.api.chat.responses import (
    ChatCompletionStreamResponseBuilder,
    annotations_from_message,
    chat_completion_response,
)
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest
from langgraph_openai_serve.api.tools import encode_function_calls
from langgraph_openai_serve.core.errors import GraphError, InvalidRequestError
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.run import GraphRun, prepare_run
from langgraph_openai_serve.graph.runner import collect_run, stream_run

logger = get_logger(__name__)


async def prepare_completion_run(
    request: ChatCompletionRequest,
    graph_registry: GraphRegistry,
) -> GraphRun:
    """Validate a Chat request and prepare its graph run."""
    graph_config = graph_registry.get_graph(request.model)
    if graph_config.supports(GraphFeature.INTERRUPTS):
        message = (
            f"Model '{request.model}' requires interrupts, which is only "
            "supported via the Responses API (/v1/responses)."
        )
        raise InvalidRequestError(message, param="model")
    graph_request, messages = decode_chat_request(request)
    return await prepare_run(graph_request, messages, graph_registry)


async def generate_completion(
    chat_request: ChatCompletionRequest, run: GraphRun
) -> ChatCompletion:
    """Generate a chat completion."""
    async with run:
        output = await collect_run(run)
    if not isinstance(output, AIMessage):
        msg = "The graph returned an unsupported Chat Completions output."
        raise GraphError(msg)
    return chat_completion_response(model=chat_request.model, message=output)


async def stream_completion(
    chat_request: ChatCompletionRequest, run: GraphRun
) -> AsyncGenerator[str, None]:
    """
    Stream a chat completion response.

    Yields:
        String chunks representing Server-Sent Events.

    """
    response_builder = ChatCompletionStreamResponseBuilder(
        chat_request.model,
        include_usage=bool(
            chat_request.stream_options and chat_request.stream_options.include_usage
        ),
    )
    chunks = _generate_stream_chunks(response_builder, run)
    try:
        async with aclosing(chunks):
            async for chunk in chunks:
                yield chunk
    except Exception:
        logger.exception("chat_completion.stream_failed")
        yield response_builder.error("Internal server error")
        yield response_builder.done()


async def _generate_stream_chunks(
    response_builder: ChatCompletionStreamResponseBuilder,
    run: GraphRun,
) -> AsyncGenerator[str, None]:
    async with run:
        yield response_builder.role()

        final_message: AIMessage | None = None
        text_parts: list[str] = []
        run_stream = stream_run(run)
        async with aclosing(run_stream):
            async for event in run_stream:
                if isinstance(event, AIMessage):
                    final_message = event
                elif isinstance(event, str):
                    text_parts.append(event)
                    yield response_builder.text(event)

    if final_message is None:
        msg = "The graph returned an unsupported Chat Completions output."
        raise GraphError(msg)
    for chunk in _final_chunks(
        response_builder,
        final_message,
        streamed_text="".join(text_parts) if text_parts else None,
    ):
        yield chunk


def _final_chunks(
    response_builder: ChatCompletionStreamResponseBuilder,
    message: AIMessage,
    *,
    streamed_text: str | None,
) -> Iterator[str]:
    message_text = str(message.text)
    if streamed_text is None:
        if message_text:
            yield response_builder.text(message_text)
    elif streamed_text != message_text:
        msg = "Streamed assistant text did not match the final assistant message."
        raise GraphError(msg)
    calls = encode_function_calls(message)
    if calls:
        yield response_builder.tool_calls(calls)
    yield response_builder.finish(
        "tool_calls" if calls else "stop",
        annotations=annotations_from_message(message),
    )
    if response_builder.include_usage and message.usage_metadata is not None:
        yield response_builder.usage(message.usage_metadata)
    yield response_builder.done()
