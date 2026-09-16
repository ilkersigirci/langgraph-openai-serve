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
from langgraph_openai_serve.api.responses.output import (
    UnsupportedResponsesOutputError,
    response_usage,
)
from langgraph_openai_serve.api.responses.request import (
    UnsupportedResponsesRequestError,
    decode_responses_request,
    selected_server_tools,
    validate_tools,
)
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.events import parse_status_event
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.graph_registry import GraphRegistry
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.runner import invoke_run, stream_run
from langgraph_openai_serve.graph.utils import GraphRun, prepare_run

logger = get_logger(__name__)


async def prepare_response_run(
    request: ResponseCreateRequest,
    graph_registry: GraphRegistry,
    *,
    checkpoint_scope: str,
) -> GraphRun:
    """Validate a Responses request and prepare its graph run."""
    graph_config = graph_registry.get_graph(request.model)
    validate_tools(request, graph_config.server_tools)
    if request.previous_response_id is not None and not graph_config.supports(
        GraphFeature.INTERRUPTS
    ):
        message = (
            "Previous response state is not supported for model "
            f"'{request.model}'; only interruptible graphs support "
            "'previous_response_id'."
        )
        raise UnsupportedResponsesRequestError(
            message,
            param="previous_response_id",
        )
    graph_request, messages, resume = decode_responses_request(
        request,
        graph_config.server_tools,
    )
    return await prepare_run(
        graph_request,
        messages,
        graph_registry,
        resume=resume,
        checkpoint_scope=checkpoint_scope,
    )


async def collect_response(request: ResponseCreateRequest, run: GraphRun) -> Response:
    """Build one non-streaming Response from the graph's durable output."""
    try:
        server_tools = selected_server_tools(request, run.config.server_tools)
        builder = ResponsesEventBuilder(
            request,
            run_id=run.run_id,
            server_tools=server_tools,
        )
    except BaseException as exc:
        run.record_failure(exc)
        await run.aclose()
        raise

    if not server_tools:
        async with run:
            output = await invoke_run(run)
            for event in _terminal_events(builder, output, run):
                if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                    return event.response
    else:
        events = _successful_response_events(
            builder,
            run,
            stream_updates=True,
            streaming=False,
        )
        async with aclosing(events):
            async for event in events:
                if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                    return event.response
    msg = "Graph execution completed without a final Response."
    raise UnsupportedResponsesOutputError(msg)


async def stream_response(
    request: ResponseCreateRequest,
    run: GraphRun,
) -> AsyncGenerator[str, None]:
    """
    Stream one prepared graph run as a typed Responses lifecycle.

    Yields:
        Named, compact Responses SSE frames.

    """
    server_tools = selected_server_tools(request, run.config.server_tools)
    builder = ResponsesEventBuilder(
        request,
        run_id=run.run_id,
        server_tools=server_tools,
    )
    events = _successful_response_events(
        builder,
        run,
        stream_updates=bool(server_tools),
    )
    try:
        async with aclosing(events):
            async for event in events:
                yield encode_event(event)
    except Exception:
        logger.exception("responses.stream_failed")
        for response_event in builder.failure("Internal server error"):
            yield encode_event(response_event)


async def _successful_response_events(
    builder: ResponsesEventBuilder,
    run: GraphRun,
    *,
    stream_updates: bool,
    streaming: bool = True,
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """
    Adapt one successful graph stream to typed Responses events.

    Yields:
        The successful Response lifecycle.

    """
    final_output: AIMessage | LangGraphInterruptBatch | None = None
    async with run:
        yield builder.created()
        yield builder.in_progress()

        expose_status = streaming and run.config.supports(GraphFeature.CLIENT_EVENTS)
        run_events = stream_run(
            run,
            stream_messages=streaming,
            stream_updates=stream_updates,
        )
        async with aclosing(run_events):
            async for graph_event in run_events:
                if isinstance(graph_event, (AIMessage, LangGraphInterruptBatch)):
                    final_output = graph_event
                    continue
                for event in _graph_response_events(
                    builder,
                    graph_event,
                    expose_status=expose_status,
                ):
                    yield event

    for event in _terminal_events(builder, final_output, run):
        yield event


def _terminal_events(
    builder: ResponsesEventBuilder,
    output: AIMessage | LangGraphInterruptBatch | None,
    run: GraphRun,
) -> Iterator[ResponseStreamEvent]:
    if isinstance(output, LangGraphInterruptBatch):
        yield from builder.finish_interrupt(
            output,
            usage=response_usage(run.usage_metadata()),
        )
        return
    if output is None:
        msg = "LangGraph stream completed without a final assistant message."
        raise RuntimeError(msg)
    yield from builder.finish(output)


def _graph_response_events(
    builder: ResponsesEventBuilder,
    event: str | CustomStreamPart | UpdatesStreamPart,
    *,
    expose_status: bool,
) -> Iterator[ResponseStreamEvent]:
    """
    Translate one non-final graph event.

    Yields:
        Zero or more typed Responses events.

    """
    if isinstance(event, str):
        yield from builder.final_delta(event)
        return
    if event["type"] == "updates":
        yield from builder.server_tools(event)
        return
    if not expose_status:
        return

    status_data = parse_status_event(event["data"])
    if status_data is None or status_data.hidden:
        return
    yield from builder.commentary(status_data.description)


__all__ = ["collect_response", "prepare_response_run", "stream_response"]
