"""Text-only OpenAI messages from Chainlit's native chat context."""

from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from openai.types.chat import ChatCompletionMessageParam

# Persisted Chainlit metadata flag that keeps UI-only messages out of model context.
MODEL_CONTEXT_EXCLUDED_KEY = "langgraph_openai_serve.exclude_from_model_context"


def mark_model_context_excluded(
    message: cl.Message | cl.AskActionMessage,
) -> None:
    """Keep a UI-only message out of subsequent model requests."""
    message.metadata = {
        **(message.metadata or {}),
        MODEL_CONTEXT_EXCLUDED_KEY: True,
    }


def mark_persisted_errors_excluded(thread: ThreadDict) -> None:
    """Preserve Chainlit's persisted error flag through native context restore."""
    for step in thread.get("steps", []):
        if "message" not in step.get("type", "") or not step.get("isError"):
            continue
        metadata = step.get("metadata")
        step["metadata"] = {
            **(metadata if isinstance(metadata, dict) else {}),
            MODEL_CONTEXT_EXCLUDED_KEY: True,
        }


def text_only_chat_messages() -> list[ChatCompletionMessageParam]:
    """Return Chainlit's role/content projection of the current chat.

    This is UI transcript context, not a lossless OpenAI protocol ledger. Chainlit's
    native projection does not retain fields such as ``tool_calls`` or
    ``tool_call_id``. Messages explicitly marked as UI-only, including failed or
    cancelled assistant output, are omitted from model context.
    """
    chainlit_messages = cl.chat_context.get()
    openai_messages = cl.chat_context.to_openai()
    return [
        cast(ChatCompletionMessageParam, openai_message)
        for chainlit_message, openai_message in zip(
            chainlit_messages,
            openai_messages,
            strict=True,
        )
        if not chainlit_message.is_error
        and not (chainlit_message.metadata or {}).get(MODEL_CONTEXT_EXCLUDED_KEY)
    ]
