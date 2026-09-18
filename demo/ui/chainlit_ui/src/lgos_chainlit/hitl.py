"""Chainlit UI for the LangGraph interrupt demo graph."""

import asyncio
import json
import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import chainlit as cl
from chainlit.context import context as chainlit_context
from chainlit.types import ThreadDict
from chainlit_utils.auth import authenticated_user_identifier
from chainlit_utils.chat.history import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)
from chainlit_utils.chat.hitl import (
    PendingHitl,
    complete_pending_hitl,
    persist_pending_hitl,
    remove_persisted_custom_elements,
    resolve_hitl,
    restore_pending_hitl,
)
from chainlit_utils.chat.resume import (
    reuse_persisted_step,
    schedule_after_thread_hydration,
)
from chainlit_utils.openai.hitl import HitlLedgerCodec, InvalidHitlLedgerError
from chainlit_utils.openai.responses import (
    final_answer,
    raise_for_response,
    response_input,
)
from chainlit_utils.openai.tools import function_calls
from openai import OpenAIError
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)

from lgos_chainlit.clients import (
    model_request,
    openai_client,
    retrieve_model,
)
from lgos_chainlit.conversation import (
    LIMITED_FUNCTIONALITY_MESSAGE,
    conversation_metadata,
    send_limited_functionality_warning,
)
from lgos_chainlit.files import (
    file_upload_overrides,
    with_response_file_parts,
)
from lgos_chainlit.lgos_protocol import (
    INTERRUPT_TOOL_NAME,
    GraphFeature,
    model_extension,
)
from lgos_chainlit.settings import settings

logger = logging.getLogger(__name__)

INTERRUPT_ELEMENT_NAME = "InterruptReview"
HITL_LEDGER_METADATA_KEY = "lgos_chainlit.hitl_interrupt_ledger"
PENDING_HITL_SESSION_KEY = "lgos_chainlit.pending_hitl_interrupt"
hitl_ledger_codec = HitlLedgerCodec(INTERRUPT_TOOL_NAME)


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    try:
        model = await retrieve_model(settings.HITL_MODEL)
    except OpenAIError:
        model = None
    extension = model_extension(model) if model is not None else None
    if (
        extension is not None
        and GraphFeature.INTERRUPTS.value not in extension.features
    ):
        msg = (
            f"The configured model {settings.HITL_MODEL!r} does not advertise "
            "interrupt support."
        )
        raise RuntimeError(msg)
    return [
        cl.ChatProfile(
            name=settings.HITL_MODEL,
            markdown_description=(
                extension.description
                if extension is not None
                else LIMITED_FUNCTIONALITY_MESSAGE
            ),
            config_overrides=file_upload_overrides(model),
        )
    ]


@cl.set_starters
async def set_starters(_current_user: cl.User | None = None) -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Human review",
            message="Refund order ORDER-123 for the customer.",
        )
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    await _warn_if_model_metadata_is_missing()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Cancel the live prompt; its durable ledger is restored on reconnect."""
    task = chainlit_context.session.current_task
    if task is not None and task is not asyncio.current_task() and not task.done():
        task.cancel()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Restore the latest durable ledger and reopen its interrupt prompt."""
    mark_persisted_errors_excluded(thread)
    cl.user_session.set(PENDING_HITL_SESSION_KEY, None)
    try:
        ledger = restore_pending_hitl(
            thread,
            codec=hitl_ledger_codec,
            metadata_key=HITL_LEDGER_METADATA_KEY,
        )
        if ledger is not None:
            await remove_persisted_custom_elements(
                thread,
                step_id=ledger.message.id,
                element_name=INTERRUPT_ELEMENT_NAME,
            )
            cl.user_session.set(PENDING_HITL_SESSION_KEY, ledger)
            schedule_after_thread_hydration(partial(reopen_pending_interrupt, ledger))
    except InvalidHitlLedgerError as exc:
        logger.exception("Persisted Chainlit HITL ledger is invalid")
        schedule_after_thread_hydration(
            partial(send_ui_message, f"Response failed: {exc}")
        )
    except Exception as exc:
        logger.exception("Chainlit HITL resume failed: %s", exc)


async def reopen_pending_interrupt(ledger: PendingHitl) -> None:
    """Recreate the live input controls from a durable interrupt ledger."""
    if cl.user_session.get(PENDING_HITL_SESSION_KEY) is not ledger:
        return
    task_started = False
    try:
        await chainlit_context.emitter.task_start()
        task_started = True
        await resolve_interrupts(ledger)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Chainlit HITL automatic resume failed")
        await send_ui_message(f"Response failed: {exc}")
    finally:
        if task_started:
            await chainlit_context.emitter.task_end()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Reply from chat context; Chainlit adds the user message before this hook."""
    try:
        await handle_message(message)
    except Exception as exc:
        logger.exception("Chainlit HITL completion failed")
        await send_ui_message(f"Response failed: {exc}")


async def handle_message(trigger_message: cl.Message | None = None) -> None:
    """
    Start a run and publish every interrupt ledger before prompting.

    Chainlit exposes public ``Message.metadata`` in restored ``ThreadDict``
    values. A model-context-excluded message therefore owns the exact Responses
    function-call ledger without private data-layer access.
    """
    pending = cl.user_session.get(PENDING_HITL_SESSION_KEY)
    if isinstance(pending, PendingHitl):
        if trigger_message is not None:
            mark_model_context_excluded(trigger_message)
            await trigger_message.update()
        await send_ui_message(
            "Resolve the pending interrupt before starting another request."
        )
        await resolve_interrupts(pending)
        return

    input_items = response_input(text_only_chat_messages())
    if trigger_message is not None:
        input_items = await with_response_file_parts(input_items, trigger_message)
    model_id = selected_model_id()

    response = await create_response(input_items, model_id=model_id)
    pending = await publish_response(response, model_id=model_id)
    if pending is not None:
        await resolve_interrupts(pending)


async def resolve_interrupts(pending: PendingHitl) -> None:
    """Resolve complete interrupt batches until the graph returns terminal text."""
    await resolve_hitl(
        pending,
        ask=ask_for_resume,
        continue_response=create_response,
        publish_response=publish_response,
    )


async def publish_response(
    response: Response,
    *,
    model_id: str,
    ledger_message: cl.Message | None = None,
) -> PendingHitl | None:
    """Persist a paused Response before prompting, or publish its final answer."""
    raise_for_response(response)
    calls = function_calls(response)
    if calls:
        return await persist_pending_hitl(
            codec=hitl_ledger_codec,
            ledger_message=ledger_message,
            model_id=model_id,
            response_id=response.id,
            function_calls=calls,
            prompt=pending_interrupt_prompt(calls),
            metadata_key=HITL_LEDGER_METADATA_KEY,
            session_key=PENDING_HITL_SESSION_KEY,
        )
    if ledger_message is not None:
        await complete_pending_hitl(
            ledger_message,
            codec=hitl_ledger_codec,
            metadata_key=HITL_LEDGER_METADATA_KEY,
            session_key=PENDING_HITL_SESSION_KEY,
        )
    await cl.Message(content=final_answer(response)).send()
    return None


async def create_response(
    input_items: list[dict[str, Any]],
    *,
    model_id: str | None = None,
    previous_response_id: str | None = None,
) -> Response:
    return await openai_client.responses.create(
        **model_request(model_id or selected_model_id()),
        input=cast("ResponseInputParam", input_items),
        previous_response_id=previous_response_id,
        store=False,
        user=authenticated_user_identifier(),
        metadata=conversation_metadata(),
    )


def selected_model_id() -> str:
    return cl.user_session.get("chat_profile") or settings.HITL_MODEL


async def _warn_if_model_metadata_is_missing() -> None:
    """Warn without blocking standard Responses behavior."""
    model_id = selected_model_id()
    try:
        model = await retrieve_model(model_id)
    except OpenAIError:
        model = None
    if model is None or model_extension(model) is None:
        await send_limited_functionality_warning()


async def ask_for_resume(
    tool_call: ResponseFunctionToolCall,
    ledger_message: cl.Message,
) -> str | None:
    try:
        payload = interrupt_payload(tool_call)
    except ValueError:
        await send_ui_message("Received an unsupported interrupt payload.")
        return None

    choices = interrupt_choices(payload)
    prompt = interrupt_prompt(payload)
    element = cl.CustomElement(
        name=INTERRUPT_ELEMENT_NAME,
        display="inline",
        props=interrupt_element_props(payload, choices),
    )
    element_message = cl.AskElementMessage(
        content=prompt,
        element=element,
        timeout=300,
    )
    # Chainlit persists ask messages but not their live element controls. Reusing
    # the ledger step identity lets a resumed prompt receive a fresh element
    # without adding another persisted message on every reconnect.
    reuse_persisted_step(element_message, ledger_message)
    ledger_message.content = element_message.content
    response = await element_message.send()
    ledger_message.content = element_message.content

    if not response:
        await send_ui_message("Interrupt input timed out.")
        return None

    if not isinstance(response, dict) or response.get("submitted") is not True:
        await send_ui_message("Interrupt was cancelled.")
        return None

    raw_decision = response.get("resume")
    if not isinstance(raw_decision, str) or not raw_decision.strip():
        await send_ui_message("No interrupt response was received.")
        return None

    decision = raw_decision.strip()
    if choices is not None:
        values, allow_other = choices
        if decision not in values and not allow_other:
            await send_ui_message("No interrupt response was received.")
            return None
    return decision


def interrupt_element_props(
    payload: object,
    choices: tuple[list[str], bool] | None,
) -> dict[str, object]:
    if choices is None:
        return {
            "prompt": interrupt_prompt(payload),
            "choices": [],
            "allow_other": True,
        }

    values, allow_other = choices
    return {
        "prompt": interrupt_prompt(payload),
        "choices": values,
        "allow_other": allow_other,
    }


def interrupt_choices(payload: object) -> tuple[list[str], bool] | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or any(not isinstance(choice, str) or not choice for choice in choices)
    ):
        return None
    return cast(list[str], choices), payload.get("allow_other") is True


def interrupt_payload(
    tool_call: ResponseFunctionToolCall,
) -> dict[str, Any]:
    try:
        arguments = json.loads(tool_call.arguments)
    except (TypeError, ValueError) as exc:
        msg = "Interrupt tool arguments must be valid JSON."
        raise ValueError(msg) from exc

    if not isinstance(arguments, dict):
        msg = "Interrupt tool arguments must be a JSON object."
        raise ValueError(msg)
    return arguments


def pending_interrupt_prompt(calls: Sequence[ResponseFunctionToolCall]) -> str:
    """Render the ledger step without letting malformed payloads skip persistence."""
    if not calls:
        return ""
    try:
        return interrupt_prompt(interrupt_payload(calls[0]))
    except ValueError:
        return ""


def interrupt_prompt(payload: object) -> str:
    if not isinstance(payload, dict):
        return _json_payload_text(payload)

    lines = [str(payload.get("question") or "Human input required.")]
    if payload.get("request"):
        lines.append(f"Request: {payload['request']}")
    else:
        details = {
            key: value
            for key, value in payload.items()
            if key not in {"question", "choices", "allow_other"}
        }
        if details:
            lines.append(_json_payload_text(details))

    return "\n\n".join(lines)


def _json_payload_text(payload: object) -> str:
    if isinstance(payload, str) and payload:
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)
