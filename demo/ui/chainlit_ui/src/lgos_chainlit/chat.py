"""Responses API Chainlit UI for the demo LangGraph server."""

import asyncio
import logging
import uuid
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
from chainlit_utils.openai.audio import (
    SPEECH_ACTION_NAME,
    add_dictation_chunk,
    start_dictation,
)
from chainlit_utils.openai.responses import (
    CommentaryTaskList,
    citation_elements,
    final_answer,
    raise_for_response,
    response_input,
)
from chainlit_utils.openai.tools import continuation_input, function_calls
from openai.types.responses import Response, ResponseInputParam

from lgos_chainlit.audio import end_dictation, read_aloud, send_speech_button
from lgos_chainlit.chat_settings import (
    background_enabled,
    chat_settings_metadata,
    configure_chat_settings,
    response_tools,
    streaming_enabled,
)
from lgos_chainlit.clients import bifrost_model, gateway, list_models, openai_client
from lgos_chainlit.conversation import (
    LIMITED_FUNCTIONALITY_MESSAGE,
    conversation_metadata,
)
from lgos_chainlit.display_files import DISPLAY_FILE_TOOL_NAME, display_file
from lgos_chainlit.files import file_upload_overrides, with_response_file_parts
from lgos_chainlit.interrupts import INTERRUPT_ACTION_NAME, interrupt_review
from lgos_chainlit.lgos_protocol import INTERRUPT_TOOL_NAME, model_description
from lgos_chainlit.mcp import mcp_tools

logger = logging.getLogger(__name__)


async def _continue_interrupt_response(
    input_items: list[dict[str, Any]],
    *,
    model_id: str,
    previous_response_id: str,
) -> Response:
    if not background_enabled():
        return await openai_client.responses.create(
            model=model_id,
            input=cast("ResponseInputParam", input_items),
            previous_response_id=previous_response_id,
            store=False,
            tools=response_tools(),
            user=authenticated_user_identifier(),
            metadata=_response_metadata(),
        )
    commentary_tasks = CommentaryTaskList()
    try:
        response = await _background_response(
            input_items,
            model=model_id,
            provider_routing=gateway.provider_routing,
            user=authenticated_user_identifier(),
            metadata=_response_metadata(),
            commentary_tasks=commentary_tasks,
            previous_response_id=previous_response_id,
        )
    except BaseException:
        await commentary_tasks.stop()
        raise
    await commentary_tasks.complete()
    return response


async def _publish_interrupt_final(response: Response) -> None:
    answer = await cl.Message(
        content=final_answer(response),
        elements=cast("list[Any]", citation_elements(response)),
    ).send()
    await send_speech_button(answer)


interrupt_form = HumanReviewForm(interrupt_review)
interrupt_workflow = HitlWorkflow(
    INTERRUPT_TOOL_NAME,
    action_name=INTERRUPT_ACTION_NAME,
    continue_response=_continue_interrupt_response,
    element_name=HUMAN_REVIEW_ELEMENT_NAME,
    prompt=interrupt_form.prompt,
    publish_final=_publish_interrupt_final,
    review=interrupt_form.props,
    validate_outputs=interrupt_form.validate_outputs,
)


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description=(
                model_description(model) or LIMITED_FUNCTIONALITY_MESSAGE
            ),
            config_overrides=file_upload_overrides(model),
        )
        for model in await list_models()
    ]


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
    answer = await _reply(message)
    if answer is not None:
        await send_speech_button(answer)


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


async def _reply(message: cl.Message) -> cl.Message | None:
    """Reply through Responses unless the thread awaits human review."""
    try:
        if await interrupt_workflow.block_new_message(message):
            return None
    except Exception as exc:
        logger.exception("Chainlit HITL state check failed")
        await send_ui_message(f"Response failed: {exc}")
        return None

    model = cl.user_session.get("chat_profile")
    if not isinstance(model, str) or not model:
        await send_ui_message("Response failed: no model profile is selected.")
        return None
    return await _response_message(message, model)


async def _response_message(message: cl.Message, model: str) -> cl.Message | None:
    """Render one Responses turn and return its answer once it completes."""
    assistant_message = cl.Message(content="")
    commentary_tasks = CommentaryTaskList()
    try:
        input_items = response_input(text_only_chat_messages())
        input_items = await with_response_file_parts(input_items, message)
        background = background_enabled()
        streaming = not background and streaming_enabled()
        metadata = _response_metadata()
        user = authenticated_user_identifier()

        while True:
            if streaming:
                response = await _stream_response(
                    input_items,
                    assistant_message,
                    model=model,
                    user=user,
                    metadata=metadata,
                    commentary_tasks=commentary_tasks,
                )
            elif background:
                response = await _background_response(
                    input_items,
                    model=model,
                    provider_routing=gateway.provider_routing,
                    user=user,
                    metadata=metadata,
                    commentary_tasks=commentary_tasks,
                )
            else:
                response = await openai_client.responses.create(
                    model=model,
                    input=cast("ResponseInputParam", input_items),
                    store=False,
                    tools=response_tools(),
                    user=user,
                    metadata=metadata,
                )

            calls = function_calls(response)
            if any(call.name == INTERRUPT_TOOL_NAME for call in calls):
                await interrupt_workflow.publish(response, model_id=model)
                await commentary_tasks.complete()
                return None

            raise_for_response(response)
            assistant_message.elements.extend(
                cast("list[Any]", citation_elements(response))
            )
            if not streaming:
                assistant_message.content += final_answer(response)
            if not calls:
                if not streaming:
                    await assistant_message.send()
                elif assistant_message.content:
                    await assistant_message.update()
                await commentary_tasks.complete()
                return assistant_message

            outputs = [
                (
                    await display_file(call)
                    if call.name == DISPLAY_FILE_TOOL_NAME
                    else await mcp_tools.execute(call)
                )
                for call in calls
            ]
            input_items.extend(continuation_input(response, outputs))
    except asyncio.CancelledError:
        await commentary_tasks.stop()
        if assistant_message.content:
            mark_model_context_excluded(assistant_message)
            await assistant_message.update()
        raise
    except Exception as exc:
        await commentary_tasks.stop()
        error = f"Response failed: {exc}"
        if assistant_message.content:
            assistant_message.content = f"{assistant_message.content}\n\n{error}"
            mark_model_context_excluded(assistant_message)
            await assistant_message.update()
        else:
            await send_ui_message(error)
        return None


def _response_metadata() -> dict[str, str]:
    metadata = chat_settings_metadata()
    metadata.update(conversation_metadata())
    return metadata


async def _background_response(
    input_items: list[dict[str, Any]],
    *,
    model: str,
    provider_routing: bool,
    user: str,
    metadata: dict[str, str],
    commentary_tasks: CommentaryTaskList,
    previous_response_id: str | None = None,
) -> Response:
    """Create and poll one background Response with best-effort cancellation."""
    client = openai_client.with_options(max_retries=2)
    idempotency_key = str(uuid.uuid4())
    create_options: dict[str, Any] = {}
    lifecycle_options: dict[str, Any] = {}
    if provider_routing:
        create_options["extra_headers"] = {"Idempotency-Key": idempotency_key}
        # Retrieve and cancel carry no model; without this query parameter
        # Bifrost routes them to its built-in openai provider.
        lifecycle_options["extra_query"] = {"provider": bifrost_model(model)[0]}
    else:
        create_options["extra_body"] = {
            "extra_headers": {"Idempotency-Key": idempotency_key}
        }
    if previous_response_id is not None:
        create_options["previous_response_id"] = previous_response_id
    response = await client.responses.create(
        model=model,
        input=cast("ResponseInputParam", input_items),
        background=True,
        store=True,
        tools=response_tools(),
        user=user,
        metadata=metadata,
        **create_options,
    )
    previous_status = None
    try:
        while response.status in {"queued", "in_progress"}:
            if response.status != previous_status:
                await commentary_tasks.add(
                    f"Background response {response.status.replace('_', ' ')}"
                )
                previous_status = response.status
            await asyncio.sleep(1)
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
    input_items: list[dict[str, Any]],
    assistant_message: cl.Message,
    *,
    model: str,
    user: str,
    metadata: dict[str, str],
    commentary_tasks: CommentaryTaskList,
) -> Response:
    """Render final text and commentary while retaining the terminal Response."""
    phases: dict[int, str | None] = {}
    final_text_streamed = False
    async with openai_client.responses.stream(
        model=model,
        input=cast("ResponseInputParam", input_items),
        store=False,
        tools=response_tools(),
        user=user,
        metadata=metadata,
    ) as stream:
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
                    await assistant_message.stream_token(event.delta)
                continue
            if event.type == "response.incomplete" or event.type == "response.failed":
                raise_for_response(event.response)
            if event.type == "response.output_text.done":
                if phases.get(event.output_index) == "commentary":
                    await commentary_tasks.add(event.text)
                continue
        completed = await stream.get_final_response()

    if (
        completed.status == "completed"
        and not final_text_streamed
        and (text := final_answer(completed))
    ):
        await assistant_message.stream_token(text)
    return cast("Response", completed)
