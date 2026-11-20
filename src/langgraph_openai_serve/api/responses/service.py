"""Prepare and execute graph runs for OpenAI Responses."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing

from langchain_core.messages import AIMessage
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
from langgraph_openai_serve.api.responses.output import ResponseContext, response_usage
from langgraph_openai_serve.api.responses.request import decode_responses_request
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.events import status_description
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.run import GraphRun, prepare_run
from langgraph_openai_serve.graph.runner import stream_run

logger = get_logger(__name__)

EventPublisher = Callable[[ResponseStreamEvent], Awaitable[None]]
"""Stores one Responses event of a run for its readers."""


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
    graph_request, messages, resume = decode_responses_request(
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
    context: ResponseContext | None = None,
    *,
    publish: EventPublisher | None = None,
) -> Response:
    """
    Build one Response from the graph's durable output.

    A background Response passes the identity it was accepted with. With
    ``publish``, each event is also available to readers in another process.
    """
    # Executing closes the run, but a cancel can stop the job while it awaits
    # the publish of an opening event, before execution starts.
    async with run:
        events = _response_events(
            _builder(request, run, context), run, streaming=bool(request.stream)
        )
        async with aclosing(events):
            async for event in events:
                if publish is not None:
                    await publish(event)
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
    try:
        async with aclosing(_response_events(builder, run, streaming=True)) as events:
            async for event in events:
                yield encode_event(event)
    except Exception:
        logger.exception("responses.stream_failed")
        for event in builder.failure("Internal server error"):
            yield encode_event(event)


def _builder(
    request: ResponseCreateRequest,
    run: GraphRun,
    context: ResponseContext | None = None,
) -> ResponsesEventBuilder:
    if context is None:
        run_id = run.interrupt.run_id if run.interrupt is not None else None
        context = ResponseContext.for_run(request, run_id)
    return ResponsesEventBuilder(context, run.request.server_tools)


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
    for event in builder.opening():
        yield event

    run_events = stream_run(
        run, streaming=streaming, stream_updates=bool(run.request.server_tools)
    )
    async with aclosing(run_events):
        async for part in run_events:
            if isinstance(part, AIMessage):
                events = builder.finish(part)
            elif isinstance(part, LangGraphInterruptBatch):
                events = builder.finish_interrupt(
                    part, usage=response_usage(run.usage_metadata())
                )
            elif isinstance(part, str):
                events = builder.final_delta(part)
            elif part["type"] == "updates":
                events = builder.server_tools(part)
            elif (description := status_description(part["data"])) is not None:
                events = builder.commentary(description)
            else:
                continue
            for event in events:
                yield event
