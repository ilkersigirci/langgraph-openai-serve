"""Prepare and execute graph runs for OpenAI Responses."""

from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing

from langchain_core.messages import AIMessage
from langgraph.types import CustomStreamPart, UpdatesStreamPart
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)

from langgraph_openai_serve.api.responses.events import (
    ResponsesEventBuilder,
    encode_event,
)
from langgraph_openai_serve.api.responses.output import response_usage
from langgraph_openai_serve.api.responses.request import (
    decode_graph_request,
    selected_server_tools,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.events import status_description
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.run import GraphRun, prepare_run
from langgraph_openai_serve.graph.runner import stream_run

logger = get_logger(__name__)


async def prepare_response_run(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    *,
    checkpoint_scope: str,
    run_id: str | None = None,
) -> GraphRun:
    """
    Validate a Responses request and prepare its graph run.

    A background run passes the interrupt ``run_id`` it chose at submission.
    """
    graph_request, messages, resume = decode_graph_request(
        request,
        graph_registry.get_graph(request.model),
    )
    return await prepare_run(
        graph_request,
        messages,
        graph_registry,
        resume=resume,
        run_id=run_id,
        checkpoint_scope=checkpoint_scope,
    )


async def collect_response(
    request: ResponseCreateRequest,
    run: GraphRun,
    *,
    response_id: str | None = None,
    created_at: int | None = None,
) -> Response:
    """Build one non-streaming Response from the graph's durable output."""
    async with run:
        events = _response_events(
            _builder(request, run, response_id=response_id, created_at=created_at),
            run,
            streaming=False,
        )
        async with aclosing(events):
            async for event in events:
                if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                    return event.response
    msg = "Graph execution completed without a final Response."
    raise RuntimeError(msg)


async def stream_response(
    request: ResponseCreateRequest,
    run: GraphRun,
) -> AsyncGenerator[str, None]:
    """
    Stream one prepared graph run as a typed Responses lifecycle.

    Yields:
        Named, compact Responses SSE frames.

    """
    builder = _builder(request, run)
    events = _response_events(builder, run, streaming=True)
    try:
        async with aclosing(events):
            async for event in events:
                yield encode_event(event)
    except Exception:
        logger.exception("responses.stream_failed")
        for response_event in builder.failure("Internal server error"):
            yield encode_event(response_event)


def _builder(
    request: ResponseCreateRequest,
    run: GraphRun,
    *,
    response_id: str | None = None,
    created_at: int | None = None,
) -> ResponsesEventBuilder:
    return ResponsesEventBuilder(
        request,
        run_id=run.interrupt.run_id if run.interrupt is not None else None,
        response_id=response_id,
        created_at=created_at,
        server_tools=selected_server_tools(request, run.config.server_tools),
    )


async def _response_events(
    builder: ResponsesEventBuilder,
    run: GraphRun,
    *,
    streaming: bool,
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """
    Adapt one graph run to typed Responses events.

    Terminal events follow run cleanup, so a failed checkpoint cleanup fails the
    Response instead of following a completed one.

    Yields:
        The successful Response lifecycle.

    """
    final_output: AIMessage | LangGraphInterruptBatch | None = None
    async with run:
        yield builder.created()
        yield builder.in_progress()

        run_events = stream_run(
            run,
            streaming=streaming,
            stream_updates=bool(builder.selected_server_tools),
        )
        async with aclosing(run_events):
            async for graph_event in run_events:
                if isinstance(graph_event, (AIMessage, LangGraphInterruptBatch)):
                    final_output = graph_event
                    continue
                for event in _graph_response_events(builder, graph_event):
                    yield event

    if isinstance(final_output, LangGraphInterruptBatch):
        terminal = builder.finish_interrupt(
            final_output,
            usage=response_usage(run.usage_metadata()),
        )
    elif final_output is not None:
        terminal = builder.finish(final_output)
    else:
        msg = "LangGraph stream completed without a final assistant message."
        raise RuntimeError(msg)
    for event in terminal:
        yield event


def _graph_response_events(
    builder: ResponsesEventBuilder,
    event: str | CustomStreamPart | UpdatesStreamPart,
) -> Iterator[ResponseStreamEvent]:
    """
    Translate one non-final graph event.

    Yields:
        Zero or more typed Responses events.

    """
    if isinstance(event, str):
        yield from builder.final_delta(event)
    elif event["type"] == "updates":
        yield from builder.server_tools(event)
    elif (description := status_description(event["data"])) is not None:
        yield from builder.commentary(description)


__all__ = ["collect_response", "prepare_response_run", "stream_response"]
