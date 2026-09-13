"""Assemble SDK-typed OpenAI Responses streaming events."""

import uuid
from collections.abc import AsyncGenerator, Collection, Iterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from langchain_core.messages import AIMessage
from langgraph.types import CustomStreamPart, UpdatesStreamPart
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCustomToolCall,
    ResponseCustomToolCallInputDeltaEvent,
    ResponseCustomToolCallInputDoneEvent,
    ResponseError,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    ResponseWebSearchCallCompletedEvent,
)
from openai.types.responses.response import IncompleteDetails

from langgraph_openai_serve.api.responses.request import selected_server_tools
from langgraph_openai_serve.api.responses.schemas import ResponseCreateRequest
from langgraph_openai_serve.api.responses.server_tools import (
    ServerToolItem,
    ServerToolTracker,
)
from langgraph_openai_serve.api.responses.service import (
    ResponseContext,
    UnsupportedResponsesOutputError,
    interrupt_output_items,
    response_incomplete_details,
    response_output_text,
    response_refusals,
    response_usage,
)
from langgraph_openai_serve.core.logging import get_logger
from langgraph_openai_serve.graph.events import parse_status_event
from langgraph_openai_serve.graph.features import GraphFeature
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch
from langgraph_openai_serve.graph.runner import (
    invoke_run,
    stream_run,
)
from langgraph_openai_serve.graph.utils import GraphRun

logger = get_logger(__name__)

_MessagePhase: TypeAlias = Literal["commentary", "final_answer"]


@dataclass
class _TextItem:
    id: str
    output_index: int
    phase: _MessagePhase
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


class ResponsesStreamBuilder:
    """Own stable state for one Responses SSE lifecycle."""

    def __init__(
        self,
        request: ResponseCreateRequest,
        *,
        run_id: str | None = None,
        server_tools: Collection[str] = (),
    ) -> None:
        self._context = ResponseContext.for_run(request, run_id=run_id)
        self._sequence_number = 0
        self._output: list[ResponseOutputItem] = []
        self._server_tool_tracker = ServerToolTracker(server_tools)
        self._final_item: _TextItem | None = None

    def created(self) -> ResponseCreatedEvent:
        """Create the initial response event."""
        return ResponseCreatedEvent(
            type="response.created",
            sequence_number=self._sequence(),
            response=self._response(status="in_progress"),
        )

    def in_progress(self) -> ResponseInProgressEvent:
        """Create the response in-progress event."""
        return ResponseInProgressEvent(
            type="response.in_progress",
            sequence_number=self._sequence(),
            response=self._response(status="in_progress"),
        )

    def commentary(self, text: str) -> Iterator[ResponseStreamEvent]:
        """
        Emit one complete commentary message lifecycle.

        Yields:
            Typed events for the message lifecycle.

        """
        item = self._new_text_item("commentary")
        yield from self._start_text_item(item)
        item.text_parts.append(text)
        yield self._text_delta(item, text)
        part = ResponseOutputText(
            annotations=[],
            logprobs=[],
            text=item.text,
            type="output_text",
        )
        yield from self._finish_text_item(item, part)

    def final_delta(self, delta: str) -> Iterator[ResponseStreamEvent]:
        """
        Emit one final-answer delta, opening its item if needed.

        Yields:
            Typed events that open or update the final message.

        """
        item = self._final_item
        if item is None:
            item = self._new_text_item("final_answer")
            self._final_item = item
            yield from self._start_text_item(item)
        item.text_parts.append(delta)
        yield self._text_delta(item, delta)

    def finish(self, message: AIMessage) -> Iterator[ResponseStreamEvent]:
        """
        Reconcile final text, finish its item, and complete the Response.

        Yields:
            Typed terminal events for a successful Response.

        """
        calls, item_status, incomplete_details = self._completion(message)
        yield from self._finish_answer(message, status=item_status)
        for call in calls:
            yield from self._tool_item(call.model_copy(update={"status": item_status}))
        response = self._response(
            status=item_status,
            usage=response_usage(message.usage_metadata),
            incomplete_details=incomplete_details,
        )
        if incomplete_details is not None:
            yield ResponseIncompleteEvent(
                type="response.incomplete",
                sequence_number=self._sequence(),
                response=response,
            )
        else:
            yield ResponseCompletedEvent(
                type="response.completed",
                sequence_number=self._sequence(),
                response=response,
            )

    def _completion(
        self, message: AIMessage
    ) -> tuple[
        list[ResponseFunctionToolCall],
        Literal["completed", "incomplete"],
        IncompleteDetails | None,
    ]:
        calls = self._server_tool_tracker.client_function_calls(message)
        incomplete_details = response_incomplete_details(message)
        status = "incomplete" if incomplete_details is not None else "completed"
        return calls, status, incomplete_details

    def _finish_answer(
        self, message: AIMessage, *, status: Literal["completed", "incomplete"]
    ) -> Iterator[ResponseStreamEvent]:
        refusals = response_refusals(message)
        has_text = bool(
            message.text
            or not (message.tool_calls or message.invalid_tool_calls or refusals)
        )
        item = self._final_item
        if item is None and (has_text or refusals):
            item = self._new_text_item("final_answer")
            self._final_item = item
            yield from self._start_text_item(item, with_text=has_text)
            if message.text:
                text = str(message.text)
                item.text_parts.append(text)
                yield self._text_delta(item, text)
        elif item is not None and item.text != str(message.text):
            msg = "Streamed assistant text did not match the final assistant message."
            raise RuntimeError(msg)

        if item is not None:
            yield from self._finish_text_item(
                item,
                response_output_text(message) if has_text else None,
                refusals=refusals,
                status=status,
            )

    def finish_interrupt(
        self,
        batch: LangGraphInterruptBatch,
        *,
        usage: ResponseUsage | None,
    ) -> Iterator[ResponseStreamEvent]:
        """
        Emit a durable interrupt batch and complete the Response.

        Yields:
            Typed function-call and terminal events.

        """
        self._server_tool_tracker.ensure_complete()
        if self._final_item is not None:
            yield from self._finish_text_item(
                self._final_item,
                response_output_text(AIMessage(content=self._final_item.text)),
            )
        for call in interrupt_output_items(batch, response_id=self._context.id):
            yield from self._tool_item(call)
        yield ResponseCompletedEvent(
            type="response.completed",
            sequence_number=self._sequence(),
            response=self._response(status="completed", usage=usage),
        )

    def failure(self, message: str) -> Iterator[ResponseStreamEvent]:
        """
        Emit the normative terminal failure sequence.

        Yields:
            The error and failed Response events.

        """
        # Keep the items already exposed to the client in the terminal snapshot.
        # The SDK replaces its accumulated Response with response.failed.
        item = self._final_item
        if (
            item is not None
            and getattr(self._output[item.output_index], "status", None)
            == "in_progress"
        ):
            self._output[item.output_index] = self._output[
                item.output_index
            ].model_copy(
                update={"content": [response_output_text(AIMessage(content=item.text))]}
            )
        self._output = [
            item.model_copy(update={"status": "incomplete"})
            if getattr(item, "status", None) == "in_progress"
            else item
            for item in self._output
        ]
        yield ResponseErrorEvent(
            type="error",
            sequence_number=self._sequence(),
            code="server_error",
            message=message,
            param=None,
        )
        yield ResponseFailedEvent(
            type="response.failed",
            sequence_number=self._sequence(),
            response=self._response(
                status="failed",
                error=ResponseError(code="server_error", message=message),
            ),
        )

    def _new_text_item(self, phase: _MessagePhase) -> _TextItem:
        return _TextItem(
            id=f"msg_{uuid.uuid4().hex}",
            output_index=len(self._output),
            phase=phase,
        )

    def _start_text_item(
        self, item: _TextItem, *, with_text: bool = True
    ) -> Iterator[ResponseStreamEvent]:
        message = ResponseOutputMessage(
            id=item.id,
            content=[],
            role="assistant",
            status="in_progress",
            type="message",
            phase=item.phase,
        )
        self._output.append(message)
        yield ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item=message,
        )
        if not with_text:
            return
        yield ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=0,
            part=ResponseOutputText(
                annotations=[],
                logprobs=[],
                text="",
                type="output_text",
            ),
        )

    def _text_delta(self, item: _TextItem, delta: str) -> ResponseTextDeltaEvent:
        return ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=0,
            delta=delta,
            logprobs=[],
        )

    def _finish_text_item(
        self,
        item: _TextItem,
        part: ResponseOutputText | None,
        *,
        refusals: Sequence[ResponseOutputRefusal] = (),
        status: Literal["completed", "incomplete"] = "completed",
    ) -> Iterator[ResponseStreamEvent]:
        content: list[ResponseOutputText | ResponseOutputRefusal] = []
        if part is not None:
            content.append(part)
            yield from self._finish_text_part(item, part)
        for refusal in refusals:
            yield from self._refusal_part(item, refusal, content_index=len(content))
            content.append(refusal)
        completed = ResponseOutputMessage(
            id=item.id,
            content=content,
            role="assistant",
            status=status,
            type="message",
            phase=item.phase,
        )
        self._output[item.output_index] = completed
        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item=completed,
        )

    def _finish_text_part(
        self, item: _TextItem, part: ResponseOutputText
    ) -> Iterator[ResponseStreamEvent]:
        for annotation_index, annotation in enumerate(part.annotations):
            yield ResponseOutputTextAnnotationAddedEvent(
                type="response.output_text.annotation.added",
                sequence_number=self._sequence(),
                output_index=item.output_index,
                item_id=item.id,
                content_index=0,
                annotation_index=annotation_index,
                annotation=annotation,
            )
        yield ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=0,
            text=part.text,
            logprobs=[],
        )
        yield ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=0,
            part=part,
        )

    def _refusal_part(
        self,
        item: _TextItem,
        part: ResponseOutputRefusal,
        *,
        content_index: int,
    ) -> Iterator[ResponseStreamEvent]:
        yield ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=content_index,
            part=ResponseOutputRefusal(type="refusal", refusal=""),
        )
        yield ResponseRefusalDeltaEvent(
            type="response.refusal.delta",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=content_index,
            delta=part.refusal,
        )
        yield ResponseRefusalDoneEvent(
            type="response.refusal.done",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=content_index,
            refusal=part.refusal,
        )
        yield ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=self._sequence(),
            output_index=item.output_index,
            item_id=item.id,
            content_index=content_index,
            part=part,
        )

    def _tool_item(
        self,
        completed: ResponseFunctionToolCall | ServerToolItem,
    ) -> Iterator[ResponseStreamEvent]:
        output_index = len(self._output)
        if completed.id is None:
            msg = "Responses tool items must include an id."
            raise RuntimeError(msg)

        if isinstance(completed, ResponseFunctionToolCall):
            pending = completed.model_copy(
                update={"arguments": "", "status": "in_progress"}
            )
        elif isinstance(completed, ResponseCustomToolCall):
            pending = completed.model_copy(
                update={"input": "", "status": "in_progress"}
            )
        else:
            pending = completed
        self._output.append(pending)
        yield ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=self._sequence(),
            output_index=output_index,
            item=pending,
        )
        if isinstance(completed, ResponseFunctionToolCall):
            yield ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=self._sequence(),
                output_index=output_index,
                item_id=completed.id,
                delta=completed.arguments,
            )
            yield ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=self._sequence(),
                output_index=output_index,
                item_id=completed.id,
                name=completed.name,
                arguments=completed.arguments,
            )
        elif isinstance(completed, ResponseCustomToolCall):
            yield ResponseCustomToolCallInputDeltaEvent(
                type="response.custom_tool_call_input.delta",
                sequence_number=self._sequence(),
                output_index=output_index,
                item_id=completed.id,
                delta=completed.input,
            )
            yield ResponseCustomToolCallInputDoneEvent(
                type="response.custom_tool_call_input.done",
                sequence_number=self._sequence(),
                output_index=output_index,
                item_id=completed.id,
                input=completed.input,
            )
        elif (
            isinstance(completed, ResponseFunctionWebSearch)
            and completed.status == "completed"
        ):
            yield ResponseWebSearchCallCompletedEvent(
                type="response.web_search_call.completed",
                sequence_number=self._sequence(),
                output_index=output_index,
                item_id=completed.id,
            )
        self._output[output_index] = completed
        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=self._sequence(),
            output_index=output_index,
            item=completed,
        )

    def server_tools(self, event: UpdatesStreamPart) -> Iterator[ResponseStreamEvent]:
        """
        Expose selected tool activity from one root graph update.

        Yields:
            Native application-tool item lifecycle events.

        """
        for item in self._server_tool_tracker.items(event):
            yield from self._tool_item(item)

    def _response(
        self,
        *,
        status: Literal["in_progress", "completed", "failed", "incomplete"],
        error: ResponseError | None = None,
        usage: ResponseUsage | None = None,
        incomplete_details: IncompleteDetails | None = None,
    ) -> Response:
        return self._context.response(
            status=status,
            output=self._output,
            error=error,
            usage=usage,
            incomplete_details=incomplete_details,
        )

    def _sequence(self) -> int:
        sequence_number = self._sequence_number
        self._sequence_number += 1
        return sequence_number


def encode_event(event: ResponseStreamEvent) -> str:
    """Encode one Responses event using the official named SSE framing."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


async def collect_response(request: ResponseCreateRequest, run: GraphRun) -> Response:
    """Build one non-streaming Response from the graph's durable output."""
    server_tools = selected_server_tools(request, run.config.server_tools)
    builder = ResponsesStreamBuilder(
        request,
        run_id=run.run_id,
        server_tools=server_tools,
    )
    if not server_tools:
        output = await invoke_run(run)
        for event in _finish_events(builder, output, run):
            if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                return event.response
    else:
        events = _successful_events(
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
    builder = ResponsesStreamBuilder(
        request,
        run_id=run.run_id,
        server_tools=server_tools,
    )
    events = _successful_events(builder, run, stream_updates=bool(server_tools))
    try:
        async with aclosing(events):
            async for event in events:
                yield encode_event(event)
    except Exception:
        logger.exception("responses.stream_failed")
        for response_event in builder.failure("Internal server error"):
            yield encode_event(response_event)


async def _successful_events(
    builder: ResponsesStreamBuilder,
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
    yield builder.created()
    yield builder.in_progress()

    final_output: AIMessage | LangGraphInterruptBatch | None = None
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

    for event in _finish_events(builder, final_output, run):
        yield event


def _finish_events(
    builder: ResponsesStreamBuilder,
    output: AIMessage | LangGraphInterruptBatch | None,
    run: GraphRun,
) -> Iterator[ResponseStreamEvent]:
    if isinstance(output, LangGraphInterruptBatch):
        yield from builder.finish_interrupt(
            output,
            usage=response_usage(run.usage_metadata()),
        )
    else:
        if output is None:
            msg = "LangGraph stream completed without a final assistant message."
            raise RuntimeError(msg)
        yield from builder.finish(output)


def _graph_response_events(
    builder: ResponsesStreamBuilder,
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


__all__ = [
    "ResponsesStreamBuilder",
    "collect_response",
    "encode_event",
    "stream_response",
]
