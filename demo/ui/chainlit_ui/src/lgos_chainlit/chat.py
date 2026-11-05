"""Responses API Chainlit UI for the demo LangGraph server."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit_utils.auth import authenticated_user_identifier
from chainlit_utils.chat.history import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
)
from chainlit_utils.chat.hitl import HitlWorkflow
from chainlit_utils.chat.human_review import HUMAN_REVIEW_ELEMENT_NAME, HumanReviewForm
from chainlit_utils.chat.streaming import MessageStream
from chainlit_utils.openai.audio import (
    SPEECH_ACTION_NAME,
    add_dictation_chunk,
    start_dictation,
)
from chainlit_utils.openai.files import file_upload_overrides, with_response_file_parts
from chainlit_utils.openai.responses import (
    CommentarySteps,
    CommentaryTaskList,
    citation_elements,
    final_answer,
    raise_for_response,
)
from chainlit_utils.openai.tools import continuation_input, function_calls
from openai import APIError, AsyncOpenAI, AsyncStream, Omit, omit
from openai.types.responses import Response, ResponseStreamEvent

from lgos_chainlit.audio import end_dictation, read_aloud, send_speech_button
from lgos_chainlit.chat_settings import (
    LIMITED_FUNCTIONALITY_MESSAGE,
    background_enabled,
    chat_settings_metadata,
    configure_chat_settings,
    response_tools,
    streaming_enabled,
)
from lgos_chainlit.clients import (
    gateway,
    list_models,
    responses_client,
    v1_client,
)
from lgos_chainlit.display_files import DISPLAY_FILE_TOOL_NAME, display_file
from lgos_chainlit.interrupts import INTERRUPT_ACTION_NAME, interrupt_review
from lgos_chainlit.lgos_protocol import (
    CONVERSATION_METADATA_KEY,
    FILE_INPUTS_FEATURE,
    INTERRUPT_TOOL_NAME,
    model_extension,
)
from lgos_chainlit.mcp import mcp_tools
from lgos_chainlit.settings import settings

logger = logging.getLogger(__name__)
BACKGROUND_RESUME_SECONDS = 1
# The SDK's backoff then outlasts a gateway restart, which a background run survives.
BACKGROUND_MAX_RETRIES = 7


@dataclass
class _Turn:
    """One assistant turn: its requests, client tool calls, and answer message."""

    model: str
    # The stateless transcript that client tool results continue.
    input_items: list[dict[str, Any]]
    streaming: bool = field(default_factory=streaming_enabled)
    answer: cl.Message = field(default_factory=lambda: cl.Message(content=""))
    commentary: CommentarySteps | CommentaryTaskList = field(
        default_factory=lambda: (
            CommentarySteps()
            if settings.STATUS_DISPLAY == "steps"
            else CommentaryTaskList()
        )
    )

    async def request(
        self,
        input_items: list[dict[str, Any]],
        *,
        previous_response_id: str | Omit = omit,
    ) -> Response:
        """Send one request; a pause keeps the text the graph showed before it."""
        async with self._keep_partial_answer():
            response = await _request_response(
                input_items,
                model=self.model,
                commentary=self.commentary,
                stream_to=self.answer if self.streaming else None,
                previous_response_id=previous_response_id,
            )
            raise_for_response(response)
        self.answer.elements.extend(cast("list[Any]", citation_elements(response)))
        if not self.streaming:
            self.answer.content += final_answer(response)
        if any(call.name == INTERRUPT_TOOL_NAME for call in function_calls(response)):
            # Send the text before the workflow publishes the review.
            if self.answer.content:
                await self.answer.send()
            await self.commentary.complete()
        return response

    async def answer_tool_calls(self, response: Response) -> Response:
        """Run the response's client function calls and request what follows."""
        async with self._keep_partial_answer():
            outputs = [
                (
                    await display_file(call)
                    if call.name == DISPLAY_FILE_TOOL_NAME
                    else await mcp_tools.execute(call)
                )
                for call in function_calls(response)
            ]
        self.input_items = [*self.input_items, *continuation_input(response, outputs)]
        return await self.request(self.input_items)

    async def finish(self) -> None:
        """Publish the turn's final answer."""
        await self.commentary.complete()
        # send() also ends a stream; it stamps the creation time that orders a
        # reloaded thread, which update() leaves to the data layer's later write.
        if not self.streaming or self.answer.content:
            await self.answer.send()
        await send_speech_button(self.answer)

    @asynccontextmanager
    async def _keep_partial_answer(self) -> AsyncGenerator[None, None]:
        """
        Keep text shown before Stop or a failure.

        Stopped text stays in later model context, as the user saw it; text
        from a failed request does not.
        """
        try:
            yield
        except BaseException as exc:
            await self.commentary.stop()
            if self.answer.content:
                if not isinstance(exc, asyncio.CancelledError):
                    mark_model_context_excluded(self.answer)
                await self.answer.send()
            raise


# The HITL workflow calls continue_response, then publish_final, in the task
# that runs one message or review submission.
_current_turn: ContextVar[_Turn] = ContextVar("_current_turn")


async def _continue_interrupt_response(
    input_items: list[dict[str, Any]],
    *,
    model_id: str,
    previous_response_id: str,
) -> Response:
    # The resumed run finishes on LGOS, so client tool results continue the
    # conversation as a new stateless request, which replays this review too.
    turn = _Turn(model_id, interrupt_workflow.response_input())
    _current_turn.set(turn)
    return await turn.request(input_items, previous_response_id=previous_response_id)


async def _publish_final(response: Response) -> None:
    """Answer client function calls until the graph answers or pauses."""
    turn = _current_turn.get()
    if function_calls(response):
        next_response = await turn.answer_tool_calls(response)
        await interrupt_workflow.publish(next_response, model_id=turn.model)
    else:
        await turn.finish()


interrupt_form = HumanReviewForm(interrupt_review)
interrupt_workflow = HitlWorkflow(
    INTERRUPT_TOOL_NAME,
    action_name=INTERRUPT_ACTION_NAME,
    continue_response=_continue_interrupt_response,
    element_name=HUMAN_REVIEW_ELEMENT_NAME,
    prompt=interrupt_form.prompt,
    publish_final=_publish_final,
    receipt=interrupt_form.receipt,
    review=interrupt_form.props,
    validate_outputs=interrupt_form.validate_outputs,
)


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    profiles = []
    for model in await list_models():
        extension = model_extension(model)
        profiles.append(
            cl.ChatProfile(
                name=model.id,
                markdown_description=(
                    extension.description
                    if extension is not None
                    else LIMITED_FUNCTIONALITY_MESSAGE
                ),
                config_overrides=file_upload_overrides(
                    extension is not None and FILE_INPUTS_FEATURE in extension.features
                ),
            )
        )
    return profiles


@cl.set_starters
async def set_starters(_current_user: cl.User | None = None) -> list[cl.Starter]:  # ruff: ignore[unused-async] - Chainlit awaits this callback.
    return [
        cl.Starter(
            label="About",
            message="Tell me about yourself.",
            icon="",
        ),
        cl.Starter(
            label="History",
            message="Remember that my favorite color is green.",
            icon="",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    await configure_chat_settings()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    await configure_chat_settings()
    mark_persisted_errors_excluded(thread)


@cl.action_callback(INTERRUPT_ACTION_NAME)
async def on_interrupt_submit(action: cl.Action) -> dict[str, object]:
    """Advance one persisted interrupt revision from an untrusted UI action."""
    return await interrupt_workflow.submit_action(action)


@cl.action_callback(SPEECH_ACTION_NAME)
async def on_speech_request(action: cl.Action) -> dict[str, object]:
    """Synthesize one part of an answer for the untrusted read-aloud control."""
    return await read_aloud(action)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await _reply(message)


@cl.on_audio_start
async def on_audio_start() -> bool:  # ruff: ignore[unused-async] - Chainlit awaits this callback.
    start_dictation()
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:  # ruff: ignore[unused-async] - Chainlit awaits this callback.
    add_dictation_chunk(chunk)


@cl.on_audio_end
async def on_audio_end() -> None:
    """Put the transcript in the chat input for the user to edit and send."""
    await end_dictation()


async def _reply(message: cl.Message) -> None:
    """Reply through Responses unless the thread awaits human review."""
    try:
        if await interrupt_workflow.block_new_message(message):
            return
    except Exception as exc:
        logger.exception("Chainlit HITL state check failed")
        await send_ui_message(f"Response failed: {exc}")
        return

    model = cl.user_session.get("chat_profile")
    if not isinstance(model, str) or not model:
        await send_ui_message("Response failed: no model profile is selected.")
        return
    try:
        input_items = await with_response_file_parts(
            interrupt_workflow.response_input(),
            message,
            client=v1_client,
            extra_query={"provider": gateway.files_provider},
        )
        turn = _Turn(model, input_items)
        _current_turn.set(turn)
        # A pause persists its review form; any other response reaches
        # _publish_final.
        response = await turn.request(input_items)
        await interrupt_workflow.publish(response, model_id=model)
    except Exception as exc:
        logger.exception("Responses request failed")
        await send_ui_message(f"Response failed: {exc}")


async def _request_response(
    input_items: list[dict[str, Any]],
    *,
    model: str,
    commentary: CommentarySteps | CommentaryTaskList,
    stream_to: cl.Message | None = None,
    previous_response_id: str | Omit = omit,
) -> Response:
    """Send one Responses request for the current turn in its delivery mode."""
    request: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "previous_response_id": previous_response_id,
        "tools": response_tools(),
        # OpenAI's end-user field, which replaces the deprecated `user`.
        "safety_identifier": authenticated_user_identifier(),
        "metadata": {
            **chat_settings_metadata(),
            CONVERSATION_METADATA_KEY: cl.context.session.thread_id,
        },
    }
    if background_enabled():
        return await _background_response(request, commentary, stream_to)
    if stream_to is not None:
        return await _stream_response(request, stream_to, commentary)
    return await responses_client.responses.create(**request, store=False)


async def _background_response(
    request: dict[str, Any],
    commentary: CommentarySteps | CommentaryTaskList,
    stream_to: cl.Message | None,
) -> Response:
    """Create one background Response and render it as it runs."""
    client = responses_client.with_options(max_retries=BACKGROUND_MAX_RETRIES)
    idempotency_headers = {"Idempotency-Key": str(uuid.uuid4())}
    create_options: dict[str, Any] = {"background": True, "store": True}
    lifecycle_options: dict[str, Any] = {}
    if gateway.type == "bifrost":
        create_options["extra_headers"] = idempotency_headers
        # Retrieve and cancel carry no model; without this query parameter
        # Bifrost routes them to its built-in openai provider.
        lifecycle_options["extra_query"] = {
            "provider": request["model"].partition("/")[0]
        }
    else:
        # LiteLLM drops a client Idempotency-Key header but forwards body
        # extra_headers upstream.
        create_options["extra_body"] = {"extra_headers": idempotency_headers}
    events = _background_events(
        client, {**request, **create_options}, lifecycle_options
    )
    response: Response | None = None
    answer_start = len(stream_to.content) if stream_to is not None else 0
    phases: dict[int, str | None] = {}
    async with (
        aclosing(events),
        (
            MessageStream(stream_to) if stream_to is not None else nullcontext()
        ) as message_stream,
    ):
        async for event in events:
            if isinstance(event, Response):
                response = event
            elif (
                event.type == "response.created"  # ruff: ignore[repeated-equality-comparison] - Explicit comparisons preserve discriminated-union narrowing in ty.
                or event.type == "response.in_progress"
            ):
                status = (event.response.status or "unknown").replace("_", " ")
                await commentary.add(f"Background response {status}")
            else:
                await _handle_stream_event(event, phases, message_stream, commentary)
    if response is None:
        msg = "The background stream closed before it returned a Response."
        raise RuntimeError(msg)
    if stream_to is not None:
        await _complete_streamed_answer(stream_to, answer_start, response)
    return response


async def _background_events(
    client: AsyncOpenAI,
    request: dict[str, Any],
    lifecycle_options: dict[str, Any],
) -> AsyncGenerator[ResponseStreamEvent | Response, None]:
    """
    Follow one background Response from its creation.

    A broken stream says nothing about its run, which continues, so the Response
    decides: a finished one ends this, and an active one resumes after the last
    event. A gateway that cannot resume delivers nothing, which makes this poll.
    Leaving early, as Stop does, cancels the run.

    Yields:
        The events that report progress, then the final Response, unless the stream
        closed before it named one.

    """
    stream = await client.responses.create(**request, stream=True)
    response_id: str | None = None
    cursor = -1
    response: Response | None = None
    try:  # ruff: ignore[too-many-statements-in-try-clause] - The whole lifecycle shares best-effort cancellation.
        while response is None:
            events = _until_broken(stream)
            async with aclosing(events):
                async for event in events:
                    cursor = event.sequence_number
                    if event.type == "response.created":
                        response_id = event.response.id
                    response = _terminal_response(event)
                    if response is not None:
                        break
                    yield event
            if response is None:
                if response_id is None:
                    return
                retrieved = await client.responses.retrieve(
                    response_id, **lifecycle_options
                )
                if retrieved.status in {"queued", "in_progress"}:
                    await asyncio.sleep(BACKGROUND_RESUME_SECONDS)
                    stream = await client.responses.retrieve(
                        response_id,
                        stream=True,
                        starting_after=cursor,
                        **lifecycle_options,
                    )
                else:
                    response = retrieved
    except (asyncio.CancelledError, GeneratorExit):
        if response_id is not None:
            await _cancel_background_response(client, response_id, lifecycle_options)
        raise
    yield response


async def _until_broken(
    stream: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """
    Yield a stream's events until it closes or breaks.

    Yields:
        Each event but keepalives, which are numbered apart from the events.

    """
    try:
        async with stream:
            async for event in stream:
                if event.type != "keepalive":
                    yield event
    except APIError:
        logger.warning("Background stream broke", exc_info=True)


def _terminal_response(event: ResponseStreamEvent) -> Response | None:
    if (
        event.type == "response.completed"  # ruff: ignore[repeated-equality-comparison] - Explicit comparisons preserve discriminated-union narrowing in ty.
        or event.type == "response.failed"
        or event.type == "response.incomplete"
    ):
        return event.response
    return None


async def _complete_streamed_answer(
    message: cl.Message, answer_start: int, response: Response
) -> None:
    """Show the final answer of a stream that was cut short or superseded by a rerun."""
    answer = final_answer(response)
    if answer and message.content[answer_start:] != answer:
        content = message.content[:answer_start] + answer
        await message.stream_token(content, is_sequence=True)


async def _cancel_background_response(
    client: AsyncOpenAI,
    response_id: str,
    lifecycle_options: dict[str, Any],
) -> None:
    try:
        await asyncio.shield(client.responses.cancel(response_id, **lifecycle_options))
    except Exception:
        logger.warning(
            "Background response cancellation failed for %s",
            response_id,
            exc_info=True,
        )


async def _stream_response(
    request: dict[str, Any],
    assistant_message: cl.Message,
    commentary: CommentarySteps | CommentaryTaskList,
) -> Response:
    """Render final text and commentary while retaining the terminal Response."""
    phases: dict[int, str | None] = {}
    final_text_streamed = False
    # Leaving MessageStream sends the batch not yet shown, so Stop and failures
    # keep every received delta.
    async with (
        MessageStream(assistant_message) as message_stream,
        responses_client.responses.stream(**request, store=False) as stream,
    ):
        async for event in stream:
            if await _handle_stream_event(event, phases, message_stream, commentary):
                final_text_streamed = True
        completed = await stream.get_final_response()

    if not final_text_streamed and (text := final_answer(completed)):
        await assistant_message.stream_token(text)
    return completed


async def _handle_stream_event(
    event: ResponseStreamEvent,
    phases: dict[int, str | None],
    message_stream: MessageStream | None,
    commentary: CommentarySteps | CommentaryTaskList,
) -> bool:
    """
    Render one stream event; return whether it streamed final-answer text.

    Without a message stream, only commentary and failures are rendered.
    """
    if event.type == "response.output_item.added":
        if event.item.type == "message":
            phases[event.output_index] = event.item.phase
        return False
    if (
        event.type == "response.output_text.delta"  # ruff: ignore[repeated-equality-comparison] - Explicit comparisons preserve discriminated-union narrowing in ty.
        or event.type == "response.refusal.delta"
    ):
        if (
            message_stream is not None
            and phases.get(event.output_index) != "commentary"
        ):
            await message_stream.stream_token(event.delta)
            return True
        return False
    if event.type == "response.incomplete" or event.type == "response.failed":  # ruff: ignore[repeated-equality-comparison] - Explicit comparisons preserve discriminated-union narrowing in ty.
        raise_for_response(event.response)
    if event.type == "response.output_text.done":
        if phases.get(event.output_index) == "commentary":
            await commentary.add(event.text)
        elif message_stream is not None:
            await message_stream.flush()
    elif event.type == "response.refusal.done" and message_stream is not None:
        await message_stream.flush()
    return False
