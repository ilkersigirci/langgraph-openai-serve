"""Responses API Chainlit UI for the demo LangGraph server."""

import asyncio
from typing import Any, cast

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit_utils.auth import authenticated_user_identifier
from chainlit_utils.chat import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)
from openai.types.responses import Response, ResponseInputParam

from lgos_chainlit.auth import register_auth_callback
from lgos_chainlit.lgos_protocol import model_description
from lgos_chainlit.utils.chat import LIMITED_FUNCTIONALITY_MESSAGE, session_metadata
from lgos_chainlit.utils.chat_settings import (
    chat_settings_metadata,
    configure_chat_settings,
    streaming_enabled,
)
from lgos_chainlit.utils.clients import list_models, model_request, openai_client
from lgos_chainlit.utils.files import (
    file_upload_overrides,
    with_response_file_parts,
)
from lgos_chainlit.utils.responses import (
    DISPLAY_FILE_TOOL,
    CommentaryTaskList,
    continuation_input,
    display_file,
    final_answer,
    function_calls,
    response_input,
)

register_auth_callback()


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
    mark_persisted_errors_excluded(thread)
    await configure_chat_settings()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Reply through stateless Responses replay."""
    model = cl.user_session.get("chat_profile")
    if not isinstance(model, str) or not model:
        await send_ui_message("Response failed: no model profile is selected.")
        return
    await _response_message(message, model)


async def _response_message(message: cl.Message, model: str) -> None:
    assistant_message = cl.Message(content="")
    commentary_tasks = CommentaryTaskList()
    try:
        input_items = response_input(text_only_chat_messages())
        input_items = await with_response_file_parts(input_items, message)
        streaming = streaming_enabled()
        metadata = chat_settings_metadata()
        metadata.update(session_metadata())
        model_options = model_request(model)
        upstream_model = cast(str, model_options["model"])
        extra_headers = cast(dict[str, str] | None, model_options.get("extra_headers"))
        user = authenticated_user_identifier()

        while True:
            if streaming:
                response = await _stream_response(
                    input_items,
                    assistant_message,
                    model=upstream_model,
                    extra_headers=extra_headers,
                    user=user,
                    metadata=metadata,
                    commentary_tasks=commentary_tasks,
                )
            else:
                response = await openai_client.responses.create(
                    model=upstream_model,
                    extra_headers=extra_headers,
                    input=cast("ResponseInputParam", input_items),
                    store=False,
                    tools=[DISPLAY_FILE_TOOL],
                    user=user,
                    metadata=metadata,
                )

            calls = function_calls(response)
            if not streaming:
                assistant_message.content += final_answer(response)
            if not calls:
                if not streaming:
                    await assistant_message.send()
                elif assistant_message.content:
                    await assistant_message.update()
                await commentary_tasks.complete()
                return

            outputs = [await display_file(call) for call in calls]
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


async def _stream_response(
    input_items: list[dict[str, Any]],
    assistant_message: cl.Message,
    *,
    model: str,
    extra_headers: dict[str, str] | None,
    user: str,
    metadata: dict[str, str],
    commentary_tasks: CommentaryTaskList,
) -> Response:
    """Render final text and commentary while retaining the terminal Response."""
    phases: dict[int, str | None] = {}
    async with openai_client.responses.stream(
        model=model,
        extra_headers=extra_headers,
        input=cast("ResponseInputParam", input_items),
        store=False,
        tools=[DISPLAY_FILE_TOOL],
        user=user,
        metadata=metadata,
    ) as stream:
        async for event in stream:
            if event.type == "response.output_item.added":
                item = event.item
                if item.type == "message":
                    phases[event.output_index] = item.phase
                continue
            if event.type == "response.output_text.delta":
                phase = phases.get(event.output_index)
                if phase == "final_answer":
                    await assistant_message.stream_token(event.delta)
                continue
            if event.type == "response.output_text.done":
                if phases.get(event.output_index) == "commentary":
                    await commentary_tasks.add(event.text)
                continue
        completed = await stream.get_final_response()

    if completed.status != "completed":
        detail = completed.error
        msg = detail.message if detail is not None else "Response failed."
        raise RuntimeError(msg)
    return cast("Response", completed)
