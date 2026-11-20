"""Build SDK-typed OpenAI Responses events and named SSE frames."""

import uuid
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Literal, TypeAlias

from langchain_core.messages import AIMessage
from langgraph.types import UpdatesStreamPart
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
    ResponseQueuedEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    ResponseWebSearchCallCompletedEvent,
)

from langgraph_openai_serve.api.responses.output import (
    ResponseContext,
    interrupt_output_items,
    response_function_calls,
    response_incomplete_details,
    response_output_text,
    response_refusals,
    response_usage,
)
from langgraph_openai_serve.api.responses.server_tools import (
    ServerToolItem,
    ServerToolTracker,
)
from langgraph_openai_serve.core.errors import GraphError
from langgraph_openai_serve.graph.interrupt import LangGraphInterruptBatch

_MessagePhase: TypeAlias = Literal["commentary", "final_answer"]

BACKGROUND_ACCEPTED_EVENTS = 2
"""How many events ``background_accepted_events`` returns; a worker numbers on."""


@dataclass
class _TextItem:
    message: ResponseOutputMessage
    output_index: int
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


class ResponsesEventBuilder:
    """Own stable state for one Responses event lifecycle."""

    def __init__(self, context: ResponseContext, server_tools: Collection[str]) -> None:
        self._context = context
        self._sequence = count(
            BACKGROUND_ACCEPTED_EVENTS if context.request.background else 0
        )
        self._output: list[ResponseOutputItem] = []
        self._server_tool_tracker = ServerToolTracker(server_tools)
        self._final_item: _TextItem | None = None

    def opening(self) -> Iterator[ResponseStreamEvent]:
        """
        Emit the events that start a Response.

        Yields:
            The created event, unless the API already accepted a background
            Response, then the in-progress event.

        """
        if not self._context.request.background:
            yield ResponseCreatedEvent(
                type="response.created",
                sequence_number=next(self._sequence),
                response=self._context.response(
                    output=self._output, status="in_progress"
                ),
            )
        yield ResponseInProgressEvent(
            type="response.in_progress",
            sequence_number=next(self._sequence),
            response=self._context.response(output=self._output, status="in_progress"),
        )

    def commentary(self, text: str) -> Iterator[ResponseStreamEvent]:
        """
        Emit one complete commentary message lifecycle.

        Yields:
            Typed events for the message lifecycle.

        """
        item = self._new_text_item("commentary")
        yield from self._start_text_item(item)
        yield self._text_delta(item, text)
        yield from self._finish_text_item(item, _output_text(text))

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
        incomplete_details = response_incomplete_details(message)
        item_status: Literal["completed", "incomplete"] = (
            "incomplete" if incomplete_details is not None else "completed"
        )
        calls = response_function_calls(message, status=item_status)
        self._server_tool_tracker.validate_final_calls(calls)
        yield from self._finish_answer(message, status=item_status)
        for call in calls:
            yield from self._tool_item(call)
        response = self._context.response(
            output=self._output,
            status=item_status,
            usage=response_usage(message.usage_metadata),
            incomplete_details=incomplete_details,
        )
        yield terminal_event(response, next(self._sequence))

    def _finish_answer(
        self, message: AIMessage, *, status: Literal["completed", "incomplete"]
    ) -> Iterator[ResponseStreamEvent]:
        refusals = response_refusals(message)
        text = str(message.text)
        has_text = bool(
            text or not (message.tool_calls or message.invalid_tool_calls or refusals)
        )
        item = self._final_item
        if item is None:
            if not (has_text or refusals):
                return
            item = self._new_text_item("final_answer")
            self._final_item = item
            yield from self._start_text_item(item, with_text=has_text)
            if text:
                item.text_parts.append(text)
                yield self._text_delta(item, text)
        elif item.text != text:
            msg = "Streamed assistant text did not match the final assistant message."
            raise GraphError(msg)

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
        calls = interrupt_output_items(batch)
        self._server_tool_tracker.validate_final_calls(calls)
        if self._final_item is not None:
            yield from self._finish_text_item(
                self._final_item, _output_text(self._final_item.text)
            )
        for call in calls:
            yield from self._tool_item(call)
        yield terminal_event(
            self._context.response(
                output=self._output, status="completed", usage=usage
            ),
            next(self._sequence),
        )

    def failure(self, message: str) -> Iterator[ResponseStreamEvent]:
        """
        Emit the normative terminal failure sequence.

        Yields:
            The error and failed Response events.

        """
        # The terminal snapshot keeps the items already exposed to the client,
        # with unfinished ones marked incomplete.
        for index, item in enumerate(self._output):
            if getattr(item, "status", None) != "in_progress":
                continue
            updates: dict[str, object] = {"status": "incomplete"}
            if self._final_item is not None and index == self._final_item.output_index:
                updates["content"] = [_output_text(self._final_item.text)]
            self._output[index] = item.model_copy(update=updates)
        yield ResponseErrorEvent(
            type="error",
            sequence_number=next(self._sequence),
            code="server_error",
            message=message,
            param=None,
        )
        yield terminal_event(
            self._context.response(
                output=self._output,
                status="failed",
                error=ResponseError(code="server_error", message=message),
            ),
            next(self._sequence),
        )

    def _new_text_item(self, phase: _MessagePhase) -> _TextItem:
        return _TextItem(
            message=ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                role="assistant",
                type="message",
                content=[],
                status="in_progress",
                phase=phase,
            ),
            output_index=len(self._output),
        )

    def _start_text_item(
        self, item: _TextItem, *, with_text: bool = True
    ) -> Iterator[ResponseStreamEvent]:
        self._output.append(item.message)
        yield ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item=item.message,
        )
        if not with_text:
            return
        yield ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
            content_index=0,
            part=_output_text(""),
        )

    def _text_delta(self, item: _TextItem, delta: str) -> ResponseTextDeltaEvent:
        return ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
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
        completed = item.message.model_copy(
            update={"content": content, "status": status}
        )
        self._output[item.output_index] = completed
        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item=completed,
        )

    def _finish_text_part(
        self, item: _TextItem, part: ResponseOutputText
    ) -> Iterator[ResponseStreamEvent]:
        for annotation_index, annotation in enumerate(part.annotations):
            yield ResponseOutputTextAnnotationAddedEvent(
                type="response.output_text.annotation.added",
                sequence_number=next(self._sequence),
                output_index=item.output_index,
                item_id=item.message.id,
                content_index=0,
                annotation_index=annotation_index,
                # The event declares its own annotation classes; validate one
                # from the output text annotation's wire mapping.
                annotation=annotation.model_dump(mode="json"),
            )
        yield ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
            content_index=0,
            text=part.text,
            logprobs=[],
        )
        yield ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
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
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
            content_index=content_index,
            part=ResponseOutputRefusal(type="refusal", refusal=""),
        )
        yield ResponseRefusalDeltaEvent(
            type="response.refusal.delta",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
            content_index=content_index,
            delta=part.refusal,
        )
        yield ResponseRefusalDoneEvent(
            type="response.refusal.done",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
            content_index=content_index,
            refusal=part.refusal,
        )
        yield ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=next(self._sequence),
            output_index=item.output_index,
            item_id=item.message.id,
            content_index=content_index,
            part=part,
        )

    def _tool_item(
        self,
        completed: ResponseFunctionToolCall | ServerToolItem,
    ) -> Iterator[ResponseStreamEvent]:
        output_index = len(self._output)
        item_id = completed.id
        if item_id is None:
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
            sequence_number=next(self._sequence),
            output_index=output_index,
            item=pending,
        )
        if isinstance(completed, ResponseFunctionToolCall):
            yield ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=next(self._sequence),
                output_index=output_index,
                item_id=item_id,
                delta=completed.arguments,
            )
            yield ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=next(self._sequence),
                output_index=output_index,
                item_id=item_id,
                arguments=completed.arguments,
            )
        elif isinstance(completed, ResponseCustomToolCall):
            yield ResponseCustomToolCallInputDeltaEvent(
                type="response.custom_tool_call_input.delta",
                sequence_number=next(self._sequence),
                output_index=output_index,
                item_id=item_id,
                delta=completed.input,
            )
            yield ResponseCustomToolCallInputDoneEvent(
                type="response.custom_tool_call_input.done",
                sequence_number=next(self._sequence),
                output_index=output_index,
                item_id=item_id,
                input=completed.input,
            )
        elif (
            isinstance(completed, ResponseFunctionWebSearch)
            and completed.status == "completed"
        ):
            yield ResponseWebSearchCallCompletedEvent(
                type="response.web_search_call.completed",
                sequence_number=next(self._sequence),
                output_index=output_index,
                item_id=item_id,
            )
        self._output[output_index] = completed
        yield ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=next(self._sequence),
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


def _output_text(text: str) -> ResponseOutputText:
    """Plain output text, for text that carries no citations."""
    return ResponseOutputText(
        annotations=[], logprobs=[], text=text, type="output_text"
    )


def background_accepted_events(
    response: Response,
) -> tuple[ResponseCreatedEvent, ResponseQueuedEvent]:
    """Return the events that open a background Response, before a worker starts it."""
    return (
        ResponseCreatedEvent(
            type="response.created", sequence_number=0, response=response
        ),
        ResponseQueuedEvent(
            type="response.queued", sequence_number=1, response=response
        ),
    )


def terminal_event(response: Response, sequence_number: int) -> ResponseStreamEvent:
    """
    Build the event that ends a Response stream.

    OpenAI defines no event for a cancelled Response and leaves its stream open,
    while a gateway reports a stream that ends without a terminal event as broken.
    A cancelled Response therefore arrives in ``response.failed``, with its status.
    """
    if response.status == "completed":
        return ResponseCompletedEvent(
            type="response.completed",
            sequence_number=sequence_number,
            response=response,
        )
    if response.status == "incomplete":
        return ResponseIncompleteEvent(
            type="response.incomplete",
            sequence_number=sequence_number,
            response=response,
        )
    return ResponseFailedEvent(
        type="response.failed",
        sequence_number=sequence_number,
        response=response,
    )


def encode_event(event: ResponseStreamEvent) -> str:
    """Encode one Responses event using the official named SSE framing."""
    return f"event: {event.type}\ndata: {event.model_dump_json(by_alias=True)}\n\n"
