"""Chainlit UI for the LangGraph interrupt demo graph."""

import asyncio
import json
import logging
from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)

from lgos_chainlit.auth import authenticated_user_identifier
from lgos_chainlit.lgos_protocol import (
    INTERRUPT_TOOL_NAME,
    THREAD_METADATA_KEY,
    GraphFeature,
    model_supports,
)
from lgos_chainlit.settings import settings
from lgos_chainlit.utils.chat import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)
from lgos_chainlit.utils.clients import discovery_client, inference_client

logger = logging.getLogger(__name__)


@cl.set_chat_profiles
async def set_chat_profiles(
    _current_user: cl.User | None = None,
) -> list[cl.ChatProfile]:
    model = await discovery_client.models.retrieve(settings.HITL_MODEL)
    if not model_supports(model, GraphFeature.INTERRUPTS):
        raise RuntimeError(
            "No interrupt-capable model metadata returned by model retrieval. "
            "Use LGOS directly or a documented pass-through that targets LGOS."
        )
    return [
        cl.ChatProfile(
            name=model.id,
            markdown_description="Approve or reject a LangGraph interrupt.",
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


@cl.on_chat_resume
def on_chat_resume(thread: ThreadDict) -> None:
    """Keep the hook registered so Chainlit restores the native chat context."""
    mark_persisted_errors_excluded(thread)


@cl.on_message
async def on_message(_message: cl.Message) -> None:
    """Reply from chat context; Chainlit adds the user message before this hook."""
    try:
        await handle_message()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Chainlit HITL completion failed")
        await send_ui_message(f"Chat completion failed: {exc}")


async def handle_message() -> None:
    messages = text_only_chat_messages()

    response = await create_completion(messages)
    assistant_message = response.choices[0].message
    tool_call = interrupt_tool_call(assistant_message)

    if tool_call is not None:
        messages.append(assistant_tool_call_message(assistant_message, tool_call))
        resume_value = await ask_for_resume(tool_call)
        if resume_value is None:
            return

        messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=json.dumps({"resume": resume_value}),
            )
        )
        response = await create_completion(messages)
        assistant_message = response.choices[0].message

    content = assistant_message.content or ""
    await cl.Message(content=content).send()


async def create_completion(
    messages: list[ChatCompletionMessageParam],
) -> ChatCompletion:
    return await inference_client.chat.completions.create(
        model=settings.chainlit_inference_model(
            cl.user_session.get("chat_profile") or settings.HITL_MODEL
        ),
        messages=messages,
        metadata={THREAD_METADATA_KEY: cl.context.session.thread_id},
        user=authenticated_user_identifier(),
    )


def assistant_tool_call_message(
    message: ChatCompletionMessage,
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionAssistantMessageParam:
    return ChatCompletionAssistantMessageParam(
        role=message.role,
        content=message.content,
        tool_calls=[tool_call_param(tool_call)],
    )


def tool_call_param(
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionMessageToolCallParam:
    return cast(
        ChatCompletionMessageToolCallParam,
        tool_call.model_dump(mode="json"),
    )


async def ask_for_resume(tool_call: ChatCompletionMessageToolCall) -> str | None:
    payload = interrupt_payload(tool_call)
    if payload is None:
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
    mark_model_context_excluded(action_message)
    response = await action_message.send()

    if not response:
        await send_ui_message("Approval timed out.")
        return None

    payload = response.get("payload") or {}
    return str(payload.get("resume") or "reject")


def interrupt_tool_call(
    message: ChatCompletionMessage,
) -> ChatCompletionMessageToolCall | None:
    for tool_call in message.tool_calls or []:
        if (
            isinstance(tool_call, ChatCompletionMessageToolCall)
            and tool_call.function.name == INTERRUPT_TOOL_NAME
        ):
            return tool_call
    return None


def interrupt_payload(
    tool_call: ChatCompletionMessageToolCall,
) -> dict[str, object] | None:
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        return None

    payload = arguments.get("payload")
    return payload if isinstance(payload, dict) else None


def interrupt_prompt(payload: dict[str, object]) -> str:
    lines = [str(payload.get("question") or "Approve this action?")]
    if payload.get("request"):
        lines.append(f"Request: {payload['request']}")

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        lines.append("Choices: " + ", ".join(str(choice) for choice in choices))

    return "\n\n".join(lines)
