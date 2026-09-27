"""Bind chainlit-utils dictation and read-aloud to the configured gateway."""

import chainlit as cl
from chainlit_utils.openai import audio

from lgos_chainlit.clients import v1_client
from lgos_chainlit.settings import settings


async def end_dictation() -> None:
    """Transcribe the finished recording into the chat input."""
    # main.py enables the microphone only when a transcription model is set.
    assert settings.AUDIO_STT_MODEL is not None
    await audio.end_dictation(client=v1_client, model=settings.AUDIO_STT_MODEL)


async def send_speech_button(answer: cl.Message) -> None:
    """Attach the read-aloud control when a speech model is configured."""
    if settings.AUDIO_TTS_MODEL is not None:
        await audio.send_speech_button(answer)


async def read_aloud(action: cl.Action) -> dict[str, object]:
    """Reply to the read-aloud control with one spoken part of an answer."""
    # The button is hidden without a speech model, but the action stays callable.
    if settings.AUDIO_TTS_MODEL is None:
        return {"ok": False, "error": "This answer cannot be read aloud."}
    return await audio.read_aloud(
        action,
        client=v1_client,
        model=settings.AUDIO_TTS_MODEL,
        voice=settings.AUDIO_TTS_VOICE,
    )
