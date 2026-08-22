"""Chainlit UI for the LangGraph interrupt demo graph."""

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import cast

import chainlit as cl
from chainlit.context import context as chainlit_context
from chainlit.types import ThreadDict
from chainlit_utils.auth import authenticated_user_identifier
from chainlit_utils.chat import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)
from openai import OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)

from lgos_chainlit.auth import register_auth_callback
from lgos_chainlit.lgos_protocol import (
    INTERRUPT_TOOL_NAME,
    GraphFeature,
    model_extension,
)
from lgos_chainlit.settings import settings
from lgos_chainlit.utils.chat import (
    LIMITED_FUNCTIONALITY_MESSAGE,
    send_limited_functionality_warning,
    session_metadata,
)
from lgos_chainlit.utils.clients import (
    model_request,
    openai_client,
    retrieve_model,
)
from lgos_chainlit.utils.thread_resume import (
    reuse_persisted_step,
    schedule_after_thread_hydration,
)

register_auth_callback()

logger = logging.getLogger(__name__)

INTERRUPT_LEDGER_METADATA_KEY = "lgos_chainlit.hitl_interrupt_ledger"
INTERRUPT_LEDGER_SCHEMA_VERSION = 1
PENDING_LEDGER_SESSION_KEY = "lgos_chainlit.pending_hitl_interrupt"
PENDING_LEDGER_STATUS = "pending"
COMPLETED_LEDGER_STATUS = "completed"


class InvalidInterruptLedgerError(ValueError):
    """A persisted Chainlit interrupt ledger is unsafe to resume."""


@dataclass(frozen=True)
class PendingInterruptLedger:
    """Validated pending state restored from one Chainlit message."""

    message: cl.Message
    model_id: str
    assistant_message: ChatCompletionMessage


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
        )
    ]


@cl.set_starters
async def set_starters(_current_user: cl.User | None = None) -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Approval",
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
    """Restore the latest durable ledger and reopen its approval prompt."""
    mark_persisted_errors_excluded(thread)
    cl.user_session.set(PENDING_LEDGER_SESSION_KEY, None)
    try:
        ledger = pending_interrupt_ledger(thread)
        if ledger is not None:
            cl.user_session.set(PENDING_LEDGER_SESSION_KEY, ledger)
            schedule_after_thread_hydration(partial(reopen_pending_interrupt, ledger))
    except InvalidInterruptLedgerError:
        logger.exception("Persisted Chainlit HITL ledger is invalid")
    except Exception as exc:
        logger.exception("Chainlit HITL resume failed: %s", exc)


async def reopen_pending_interrupt(ledger: PendingInterruptLedger) -> None:
    """Recreate the live approval actions from a durable interrupt ledger."""
    if cl.user_session.get(PENDING_LEDGER_SESSION_KEY) is not ledger:
        return
    task_started = False
    try:
        await chainlit_context.emitter.task_start()
        task_started = True
        await resolve_interrupts(
            assistant_message=ledger.assistant_message,
            model_id=ledger.model_id,
            ledger_message=ledger.message,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Chainlit HITL automatic resume failed")
        await send_ui_message(f"Chat completion failed: {exc}")
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
        await send_ui_message(f"Chat completion failed: {exc}")


async def handle_message(trigger_message: cl.Message | None = None) -> None:
    """
    Start a run and publish every interrupt ledger before prompting.

    Chainlit exposes public ``Message.metadata`` in restored ``ThreadDict``
    values. A model-context-excluded assistant message therefore owns the exact
    OpenAI tool-call ledger without private data-layer access.
    """
    pending = cl.user_session.get(PENDING_LEDGER_SESSION_KEY)
    if isinstance(pending, PendingInterruptLedger):
        if trigger_message is not None:
            mark_model_context_excluded(trigger_message)
            await trigger_message.update()
        await send_ui_message(
            "Resolve the pending approval before starting another request."
        )
        await resolve_interrupts(
            assistant_message=pending.assistant_message,
            model_id=pending.model_id,
            ledger_message=pending.message,
        )
        return

    messages = text_only_chat_messages()
    model_id = selected_model_id()

    response = await create_completion(messages, model_id=model_id)
    await resolve_interrupts(
        assistant_message=response.choices[0].message,
        model_id=model_id,
    )


async def resolve_interrupts(
    *,
    assistant_message: ChatCompletionMessage,
    model_id: str,
    ledger_message: cl.Message | None = None,
) -> None:
    """Resolve complete interrupt batches until the graph returns terminal text."""
    while True:
        tool_calls = interrupt_tool_calls(assistant_message)
        if tool_calls is None:
            if ledger_message is not None:
                await mark_ledger_completed(ledger_message)
            await send_ui_message("Received an unsupported tool-call batch.")
            return
        if not tool_calls:
            break

        ledger_message = await persist_pending_ledger(
            ledger_message=ledger_message,
            model_id=model_id,
            assistant_message=assistant_message,
        )
        decisions = []
        for tool_call in tool_calls:
            decision = await ask_for_resume(tool_call, ledger_message)
            if decision is None:
                return
            decisions.append(decision)

        resume_messages: list[ChatCompletionMessageParam] = [
            assistant_tool_call_message(assistant_message),
            *[
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=json.dumps({"resume": decision}),
                )
                for tool_call, decision in zip(tool_calls, decisions, strict=True)
            ],
        ]
        response = await create_completion(resume_messages, model_id=model_id)
        assistant_message = response.choices[0].message

    if ledger_message is not None:
        await mark_ledger_completed(ledger_message)
    await cl.Message(content=assistant_message.content or "").send()


async def create_completion(
    messages: list[ChatCompletionMessageParam],
    *,
    model_id: str | None = None,
) -> ChatCompletion:
    return await openai_client.chat.completions.create(
        **model_request(model_id or selected_model_id()),
        messages=messages,
        user=authenticated_user_identifier(),
        metadata=session_metadata(),
    )


def selected_model_id() -> str:
    return cl.user_session.get("chat_profile") or settings.HITL_MODEL


async def persist_pending_ledger(
    *,
    ledger_message: cl.Message | None,
    model_id: str,
    assistant_message: ChatCompletionMessage,
) -> cl.Message:
    """Create or update the one public Chainlit message that owns the ledger."""
    ledger = {
        "schema_version": INTERRUPT_LEDGER_SCHEMA_VERSION,
        "status": PENDING_LEDGER_STATUS,
        "model_id": model_id,
        "assistant_message": assistant_tool_call_message(assistant_message),
    }
    prompt = pending_interrupt_prompt(assistant_message)
    if ledger_message is None:
        ledger_message = cl.Message(content=prompt)
        set_ledger_message_metadata(ledger_message, ledger)  # ty: ignore[invalid-argument-type]
        await ledger_message.send()
    else:
        ledger_message.content = prompt
        set_ledger_message_metadata(ledger_message, ledger)  # ty: ignore[invalid-argument-type]
        await ledger_message.update()
    cl.user_session.set(
        PENDING_LEDGER_SESSION_KEY,
        PendingInterruptLedger(
            message=ledger_message,
            model_id=model_id,
            assistant_message=assistant_message,
        ),
    )
    return ledger_message


async def mark_ledger_completed(ledger_message: cl.Message) -> None:
    """Persist a terminal marker before rendering output so resume cannot replay."""
    set_ledger_message_metadata(
        ledger_message,
        {
            "schema_version": INTERRUPT_LEDGER_SCHEMA_VERSION,
            "status": COMPLETED_LEDGER_STATUS,
        },
    )
    await ledger_message.update()
    cl.user_session.set(PENDING_LEDGER_SESSION_KEY, None)


def set_ledger_message_metadata(
    message: cl.Message,
    ledger: dict[str, object],
) -> None:
    # During on_chat_resume(), Message.from_dict() shares this mapping with the
    # original thread step. Chainlit rebuilds chat context from that step after
    # the hook returns, so preserve its identity while updating the message.
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    mark_model_context_excluded(message)
    metadata.update(message.metadata or {})
    metadata[INTERRUPT_LEDGER_METADATA_KEY] = ledger
    message.metadata = metadata


def pending_interrupt_ledger(thread: ThreadDict) -> PendingInterruptLedger | None:
    """Decode the newest ledger step; a completed ledger blocks older replay."""
    for step in reversed(thread.get("steps", [])):
        metadata = step.get("metadata")
        if (
            not isinstance(metadata, dict)
            or INTERRUPT_LEDGER_METADATA_KEY not in metadata
        ):
            continue

        parsed = parse_interrupt_ledger_metadata(
            metadata[INTERRUPT_LEDGER_METADATA_KEY]
        )
        if parsed is None:
            return None
        model_id, assistant_message = parsed
        restored_step = dict(step)
        created_at = restored_step.get("createdAt")
        if isinstance(created_at, str) and not created_at.endswith("Z"):
            # The pinned SQL layer returns naive ISO text, but its write path
            # accepts only the same timestamp with an explicit UTC suffix.
            restored_step["createdAt"] = f"{created_at}Z"
        try:
            message = cl.Message.from_dict(restored_step)  # ty: ignore[invalid-argument-type]
        except (KeyError, TypeError, ValueError) as exc:
            msg = "The pending interrupt message cannot be restored."
            raise InvalidInterruptLedgerError(msg) from exc
        return PendingInterruptLedger(
            message=message,
            model_id=model_id,
            assistant_message=assistant_message,
        )
    return None


def parse_interrupt_ledger_metadata(
    raw_ledger: object,
) -> tuple[str, ChatCompletionMessage] | None:
    if not isinstance(raw_ledger, dict):
        msg = "Interrupt ledger metadata is not an object."
        raise InvalidInterruptLedgerError(msg)
    if raw_ledger.get("schema_version") != INTERRUPT_LEDGER_SCHEMA_VERSION:
        msg = "Interrupt ledger schema is unsupported."
        raise InvalidInterruptLedgerError(msg)

    status = raw_ledger.get("status")
    if status == COMPLETED_LEDGER_STATUS:
        return None
    if status != PENDING_LEDGER_STATUS:
        msg = "Interrupt ledger status is invalid."
        raise InvalidInterruptLedgerError(msg)
    model_id = raw_ledger.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        msg = "Interrupt ledger model ID is invalid."
        raise InvalidInterruptLedgerError(msg)
    try:
        assistant_message = ChatCompletionMessage.model_validate(
            raw_ledger.get("assistant_message")
        )
    except (TypeError, ValueError) as exc:
        msg = "Interrupt ledger assistant message is invalid."
        raise InvalidInterruptLedgerError(msg) from exc
    if not interrupt_tool_calls(assistant_message):
        msg = "Interrupt ledger assistant message has no interrupt calls."
        raise InvalidInterruptLedgerError(msg)
    return model_id, assistant_message


async def _warn_if_model_metadata_is_missing() -> None:
    """Warn without blocking standard Chat Completions behavior."""
    model_id = selected_model_id()
    try:
        model = await retrieve_model(model_id)
    except OpenAIError:
        model = None
    if model is None or model_extension(model) is None:
        await send_limited_functionality_warning()


def assistant_tool_call_message(
    message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
    """Preserve the complete assistant tool-call ledger without re-encoding it."""
    return ChatCompletionAssistantMessageParam(
        role=message.role,
        content=message.content,
        tool_calls=[
            tool_call_param(tool_call)  # ty: ignore[invalid-argument-type]
            for tool_call in message.tool_calls or []
        ],
    )


def tool_call_param(
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionMessageToolCallParam:
    return cast(
        ChatCompletionMessageToolCallParam,
        tool_call.model_dump(mode="json"),
    )


async def ask_for_resume(
    tool_call: ChatCompletionMessageToolCall,
    ledger_message: cl.Message,
) -> str | None:
    try:
        payload = interrupt_payload(tool_call)
    except ValueError:
        await send_ui_message("Received an unsupported interrupt payload.")
        return None

    action_message = cl.AskActionMessage(
        content=interrupt_prompt(payload),
        actions=[
            cl.Action(
                name="approve",
                label="Approve",
                icon="check",
                payload={"resume": "approve"},
            ),
            cl.Action(
                name="reject",
                label="Reject",
                icon="x",
                payload={"resume": "reject"},
            ),
        ],
        timeout=300,
    )
    # Chainlit persists ask messages but not their live actions. Reusing the
    # ledger step identity lets its resumed prompt receive a fresh ask without
    # adding another persisted message on every reconnect.
    reuse_persisted_step(action_message, ledger_message)
    ledger_message.content = action_message.content
    response = await action_message.send()
    ledger_message.content = action_message.content

    if not response:
        await send_ui_message("Approval timed out.")
        return None

    response_payload = response.get("payload")
    if not isinstance(response_payload, dict):
        await send_ui_message("No approval decision was received.")
        return None

    decision = response_payload.get("resume")
    if decision not in {"approve", "reject"}:
        await send_ui_message("No approval decision was received.")
        return None
    return decision


def interrupt_tool_calls(
    message: ChatCompletionMessage,
) -> list[ChatCompletionMessageToolCall] | None:
    tool_calls = list(message.tool_calls or [])
    if any(tool_call.function.name != INTERRUPT_TOOL_NAME for tool_call in tool_calls):  # ty: ignore[unresolved-attribute]
        return None
    return tool_calls  # ty: ignore[invalid-return-type]


def interrupt_payload(
    tool_call: ChatCompletionMessageToolCall,
) -> object:
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (TypeError, ValueError) as exc:
        msg = "Interrupt tool arguments must be valid JSON."
        raise ValueError(msg) from exc

    if not isinstance(arguments, dict):
        msg = "Interrupt tool arguments must be a JSON object."
        raise TypeError(msg)
    if "payload" not in arguments:
        msg = "Interrupt tool arguments must contain a payload."
        raise ValueError(msg)

    return arguments["payload"]


def pending_interrupt_prompt(message: ChatCompletionMessage) -> str:
    """Render the ledger step without letting malformed payloads skip persistence."""
    tool_calls = interrupt_tool_calls(message)
    if not tool_calls:
        return ""
    try:
        return interrupt_prompt(interrupt_payload(tool_calls[0]))
    except ValueError:
        return ""


def interrupt_prompt(payload: object) -> str:
    if not isinstance(payload, dict):
        return _json_payload_text(payload)

    lines = [str(payload.get("question") or "Approve this action?")]
    if payload.get("request"):
        lines.append(f"Request: {payload['request']}")
    elif set(payload) != {"question"}:
        lines.append(_json_payload_text(payload))

    return "\n\n".join(lines)


def _json_payload_text(payload: object) -> str:
    if isinstance(payload, str) and payload:
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)
