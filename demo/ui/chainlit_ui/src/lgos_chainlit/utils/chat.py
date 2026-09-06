"""LGOS-specific Chainlit messages and compatibility warning."""

import chainlit as cl

from lgos_chainlit.lgos_protocol import LGOS_EXTENSION_KEY, SESSION_ID_METADATA_KEY

LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: The configured OpenAI endpoint did not return valid "
    f"{LGOS_EXTENSION_KEY} model metadata. Runtime settings, file uploads, and "
    "interrupt profile checks may be unavailable."
)


def session_metadata() -> dict[str, str]:
    """Return the stable Chainlit thread ID as OpenAI request metadata."""
    return {SESSION_ID_METADATA_KEY: cl.context.session.thread_id}


async def send_limited_functionality_warning() -> None:
    """Show a transient warning when LGOS metadata was stripped."""
    await cl.context.emitter.send_toast(
        LIMITED_FUNCTIONALITY_MESSAGE,
        type="warning",
    )
