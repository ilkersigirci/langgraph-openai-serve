"""Responses API Chainlit UI for the demo LangGraph server."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
    text_only_chat_messages,
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
    CommentaryTaskList,
    citation_elements,
    final_answer,
    raise_for_response,
    response_input,
)
from chainlit_utils.openai.tools import continuation_input, function_calls
from openai import Omit, omit
from openai.types.responses import Response

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
    bifrost_model,
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

logger = logging.getLogger(__name__)
BACKGROUND_POLL_SECONDS = 1


@dataclass
class _Turn:
    """One assistant turn: its requests, client tool calls, and answer message."""

    model: str
    # The stateless transcript that client tool results continue.
    input_items: list[dict[str, Any]]
    streaming: bool = field(
        default_factory=lambda: not background_enabled() and streaming_enabled()
    )
    answer: cl.Message = field(default_factory=lambda: cl.Message(content=""))
    commentary_tasks: CommentaryTaskList = field(default_factory=CommentaryTaskList)

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
                commentary_tasks=self.commentary_tasks,
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
            await self.commentary_tasks.complete()
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
        await self.commentary_tasks.complete()
        # send() also ends a stream; it stamps the creation time that orders a
        # reloaded thread, which update() leaves to the data layer's later write.
        if not self.streaming or self.answer.content:
            await self.answer.send()
        await send_speech_button(self.answer)

    @asynccontextmanager
    async def _keep_partial_answer(self) -> AsyncIterator[None]:
        """Keep text shown before Stop or a failure.

        Stopped text stays in later model context, as the user saw it; text
        from a failed request does not.
        """
        try:
            yield
        except BaseException as exc:
            await self.commentary_tasks.stop()
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
    # conversation as a new stateless request.
    turn = _Turn(model_id, response_input(text_only_chat_messages()))
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
async def set_starters(_current_user: cl.User | None = None) -> list[cl.Starter]:
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
async def on_audio_start() -> bool:
    start_dictation()
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:
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
            response_input(text_only_chat_messages()),
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
        await send_ui_message(f"Response failed: {exc}")


async def _request_response(
    input_items: list[dict[str, Any]],
    *,
    model: str,
    commentary_tasks: CommentaryTaskList,
    stream_to: cl.Message | None = None,
    previous_response_id: str | Omit = omit,
) -> Response:
    """Send one Responses request for the current turn in its delivery mode."""
    request: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "previous_response_id": previous_response_id,
        "tools": response_tools(),
        "user": authenticated_user_identifier(),
        "metadata": {
            **chat_settings_metadata(),
            CONVERSATION_METADATA_KEY: cl.context.session.thread_id,
        },
    }
    if stream_to is not None:
        return await _stream_response(request, stream_to, commentary_tasks)
    if background_enabled():
        return await _background_response(request, commentary_tasks)
    return await responses_client.responses.create(**request, store=False)


async def _background_response(
    request: dict[str, Any],
    commentary_tasks: CommentaryTaskList,
) -> Response:
    """Create and poll one background Response with best-effort cancellation."""
    client = responses_client.with_options(max_retries=2)
    idempotency_headers = {"Idempotency-Key": str(uuid.uuid4())}
    create_options: dict[str, Any] = {}
    lifecycle_options: dict[str, Any] = {}
    if gateway.type == "bifrost":
        create_options["extra_headers"] = idempotency_headers
        # Retrieve and cancel carry no model; without this query parameter
        # Bifrost routes them to its built-in openai provider.
        lifecycle_options["extra_query"] = {
            "provider": bifrost_model(request["model"])[0]
        }
    else:
        create_options["extra_body"] = {"extra_headers": idempotency_headers}
    response = await client.responses.create(
        **request,
        **create_options,
        background=True,
        store=True,
    )
    previous_status = None
    try:
        while response.status in {"queued", "in_progress"}:
            if response.status != previous_status:
                await commentary_tasks.add(
                    f"Background response {response.status.replace('_', ' ')}"
                )
                previous_status = response.status
            await asyncio.sleep(BACKGROUND_POLL_SECONDS)
            response = await client.responses.retrieve(response.id, **lifecycle_options)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(
                client.responses.cancel(response.id, **lifecycle_options)
            )
        except Exception:
            logger.warning(
                "Background response cancellation failed for %s",
                response.id,
                exc_info=True,
            )
        raise
    return response


async def _stream_response(
    request: dict[str, Any],
    assistant_message: cl.Message,
    commentary_tasks: CommentaryTaskList,
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
            if event.type == "response.output_item.added":
                item = event.item
                if item.type == "message":
                    phases[event.output_index] = item.phase
                continue
            if (
                event.type == "response.output_text.delta"
                or event.type == "response.refusal.delta"
            ):
                phase = phases.get(event.output_index)
                if phase != "commentary":
                    final_text_streamed = True
                    await message_stream.stream_token(event.delta)
                continue
            if event.type == "response.incomplete" or event.type == "response.failed":
                raise_for_response(event.response)
            if event.type == "response.output_text.done":
                if phases.get(event.output_index) == "commentary":
                    await commentary_tasks.add(event.text)
                else:
                    await message_stream.flush()
                continue
            if event.type == "response.refusal.done":
                await message_stream.flush()
                continue
        completed = await stream.get_final_response()

    if (
        completed.status == "completed"
        and not final_text_streamed
        and (text := final_answer(completed))
    ):
        await assistant_message.stream_token(text)
    return completed
