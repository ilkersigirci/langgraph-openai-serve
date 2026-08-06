"""LGOS-specific Chainlit messages and compatibility warning."""

import chainlit as cl

from lgos_chainlit.lgos_protocol import LGOS_EXTENSION_KEY

LIMITED_FUNCTIONALITY_MESSAGE = (
    "Limited functionality: The configured OpenAI endpoint did not return valid "
    f"{LGOS_EXTENSION_KEY} model metadata. Runtime settings, client events, and "
    "interrupts may be unavailable."
)


async def send_limited_functionality_warning() -> None:
    """Show a transient warning when LGOS metadata was stripped."""
    await cl.context.emitter.send_toast(
        LIMITED_FUNCTIONALITY_MESSAGE,
        type="warning",
    )
